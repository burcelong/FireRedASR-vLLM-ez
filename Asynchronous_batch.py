import torch
import argparse
import os
import traceback
import threading
import time
import asyncio
import gc
import wave  # 用于计算语音时长
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from fireredasr.models.fireredasr import FireRedAsr
import tempfile
import uuid
from dataclasses import dataclass, field

# ========================= 配置 =========================
MODEL_PATH = os.getenv("FIRERED_MODEL_PATH")


# 添加安全全局对象
torch.serialization.add_safe_globals([argparse.Namespace])

# 全局：短语音阈值（秒） —— 仅在双实例模式下用于分流；单实例模式不会使用它做判断
SHORT_THRESHOLD = 4.0

# 每多少个 batch 做一次周期性 empty_cache
EMPTY_CACHE_PERIOD = 20

# 是否加载双实例（通过命令行设置，默认 True；在 __main__ 里最终赋值）
DUAL_INSTANCES: bool = True


@dataclass
class BatchRequest:
    """单个请求的数据结构（增加语音时长）"""
    request_id: str
    filename: str
    temp_path: str
    config: Dict[str, Any]
    result: Dict[str, Any]  # 存储处理结果
    event: asyncio.Event  # 异步事件
    start_time: float  # 计时用
    enqueue_time: float  # 入队时间
    audio_duration: float  # 语音时长（秒）


@dataclass
class ModelInstance:
    """
    单模型实例类：
    - 本实例只处理一个队列（短 or 长 or 单实例）
    - 使用 Condition 来等待/唤醒，避免忙等
    """
    instance_id: int
    name: str  # "short" / "long" / "single"，仅用于日志
    model: Any = None
    queue: List[BatchRequest] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)
    cond: threading.Condition = field(default_factory=lambda: threading.Condition(threading.Lock()))
    is_processing: bool = False
    process_lock: threading.Lock = field(default_factory=threading.Lock)
    batch_thread: threading.Thread = None
    batch_size: int = 16
    batch_timeout: float = 0.002  # 秒
    max_queue_size: int = 200
    batch_counter: int = 0  # 用于周期性显存清理

    def start(self):
        self.batch_thread = threading.Thread(
            target=batch_processor, args=(self,), daemon=True
        )
        self.batch_thread.start()


class ASRRequest(BaseModel):
    """ASR请求参数模型"""
    beam_size: int = 1
    nbest: int = 1
    decode_max_len: int = 0
    softmax_smoothing: float = 1
    aed_length_penalty: float = 0.6
    eos_penalty: float = 1.0
    use_gpu: int = 1  # 0=CPU，1=GPU


# 全局实例（双实例模式下两个都存在；单实例模式下仅使用 short_instance 作为“single”）
short_instance: Optional[ModelInstance] = None
long_instance: Optional[ModelInstance] = None

# 一个全局锁，用于统计两个队列总长度（仅双实例模式用）
global_len_lock = threading.Lock()

# 创建FastAPI应用
app = FastAPI(
    title="ASR服务（提前返回版）",
    description="支持 --dual 选择双实例或单实例；双实例：短/长分流；单实例：统一队列，不做时长分流"
)


def get_audio_duration(file_path: str) -> float:
    """计算语音文件时长（秒）"""
    try:
        with wave.open(file_path, 'rb') as wf:
            frames = wf.getnframes()
            rate = wf.getframerate()
            if rate == 0:
                return 0.0
            return frames / float(rate)
    except Exception as e:
        print(f"计算语音时长失败: {e}")
        # 保护：不知道时长时按长语音处理（双实例模式下会落到 long）
        return SHORT_THRESHOLD + 999.0


def safe_results_index(results, i):
    """安全获取模型返回的第 i 条结果（从 nbest 列表取第一个）"""
    if results is None or i >= len(results):
        return {"uttid": "", "text": "", "nbest": []}
    item = results[i]
    if isinstance(item, list):
        if len(item) == 0:
            return {"uttid": "", "text": "", "nbest": []}
        return item[0]
    return item


def run_inference_with_oom_recovery(instance: ModelInstance, batch_uttid, batch_wav_path, config, take_count):
    """
    推理执行，带 OOM 恢复逻辑：
    - 首次按 take_count 批量推
    - 若 OOM：empty_cache 一次，降半批量重试（最多 2 次降批）
    - 返回 (results, actually_processed_sz)
    """
    try_sizes = [take_count, max(1, take_count // 2), max(1, take_count // 4)]
    last_err = None
    for sz in try_sizes:
        try:
            uttid = batch_uttid[:sz]
            paths = batch_wav_path[:sz]
            with torch.no_grad():
                res = instance.model.transcribe(uttid, paths, config)
            return res, sz
        except RuntimeError as e:
            msg = str(e)
            last_err = e
            if "out of memory" in msg.lower():
                print(f"[WARN][{instance.name}] CUDA OOM，执行 empty_cache 并降批重试（当前 batch={sz}）")
                torch.cuda.empty_cache()
                time.sleep(0.01)
                continue
            else:
                raise
    # 如果所有重试都失败，抛出最后一次异常
    raise last_err if last_err else RuntimeError("Unknown inference error")


def batch_processor(instance: ModelInstance):
    """批处理线程（本实例只处理自身队列；支持样本提前返回）"""
    while True:
        # === 等待批处理触发 ===
        with instance.cond:
            start_wait = time.time()
            while True:
                qlen = len(instance.queue)
                if qlen >= instance.batch_size:
                    break
                remaining = instance.batch_timeout - (time.time() - start_wait)
                if qlen > 0 and remaining <= 0:
                    break
                if qlen == 0:
                    instance.cond.wait(timeout=instance.batch_timeout)
                else:
                    if remaining > 0:
                        instance.cond.wait(timeout=remaining)
                    else:
                        break

        # === 占用处理锁 ===
        with instance.process_lock:
            if instance.is_processing:
                print(f"[DEBUG][{instance.name}] 实例忙碌，跳过本次调度")
                continue
            instance.is_processing = True

        # === 组 batch ===
        current_batch: List[BatchRequest] = []
        try:
            with instance.lock:
                if instance.queue:
                    take_count = min(instance.batch_size, len(instance.queue))
                    current_batch = instance.queue[:take_count]
                    instance.queue = instance.queue[take_count:]
                    print(f"[DEBUG][{instance.name}] 取出{len(current_batch)}个任务（剩余：{len(instance.queue)}）")
        except Exception as e:
            print(f"[ERROR][{instance.name}] 取出任务失败：{e}")
            with instance.process_lock:
                instance.is_processing = False
            continue

        if not current_batch:
            with instance.process_lock:
                instance.is_processing = False
            continue

        # === 执行推理（支持提前返回） ===
        results = None
        processed_sz = 0
        leftover: List[BatchRequest] = []
        try:
            infer_start = time.time()
            batch_uttid = [req.request_id for req in current_batch]
            batch_wav_path = [req.temp_path for req in current_batch]
            base_config = current_batch[0].config  # 同批次使用相同配置（从第一个请求拷贝）

            # 为本批构造 on_finish 回调：谁完成谁返回
            def on_finish_cb(utt_idx: int, hyp: Dict[str, Any]):
                try:
                    req = current_batch[utt_idx]
                except Exception:
                    return

                # 组装统一返回体
                if "text" in hyp:
                    res_obj = {
                        "uttid": req.request_id,
                        "text": hyp["text"],
                        "score": float(hyp.get("score", 0.0)),
                        "wav": req.filename
                    }
                else:
                    y = hyp.get("yseq", [])
                    if hasattr(y, "tolist"):
                        y = y.tolist()
                    res_obj = {
                        "uttid": req.request_id,
                        "text": "",
                        "yseq": y,
                        "score": float(hyp.get("score", 0.0)),
                        "wav": req.filename
                    }

                total_process_time = time.time() - req.start_time
                req.result = {
                    "filename": req.filename,
                    "results": res_obj,
                    "status": "success",
                    "process_time": f"{total_process_time:.4f}秒",
                    "instance_id": instance.instance_id,
                    "batch_size": len(current_batch),
                    "queue_type": instance.name,
                    "audio_duration": f"{req.audio_duration:.2f}秒",
                }
                req.event.set()

                # 清理该样本临时文件
                try:
                    if os.path.exists(req.temp_path):
                        os.unlink(req.temp_path)
                except Exception as ce:
                    print(f"[WARN][{instance.name}] 清理临时文件失败: {ce}")

            # 带回调的 config
            config = dict(base_config)
            config["on_finish"] = on_finish_cb

            # 推理（带 OOM 自恢复）
            results, processed_sz = run_inference_with_oom_recovery(
                instance,
                batch_uttid=batch_uttid,
                batch_wav_path=batch_wav_path,
                config=config,
                take_count=len(current_batch),
            )

            # 如果 OOM 降批，仅处理前 processed_sz 条，其余重新入队
            if processed_sz < len(current_batch):
                leftover = current_batch[processed_sz:]
                with instance.cond:
                    instance.queue = leftover + instance.queue  # 回队列头部
                    instance.cond.notify_all()
                current_batch = current_batch[:processed_sz]

            infer_duration = time.time() - infer_start
            print(f"[INFO][{instance.name}] 批处理完成，耗时{infer_duration:.3f}秒，batch={processed_sz}{'(+requeue '+str(len(leftover))+')' if leftover else ''}")

            # 对未通过回调提前返回的样本，做兜底返回
            for i, req in enumerate(current_batch):
                if req.event.is_set():
                    continue  # 已提前返回
                res = safe_results_index(results, i)
                total_process_time = time.time() - req.start_time
                req.result = {
                    "filename": req.filename,
                    "results": res,
                    "status": "success",
                    "process_time": f"{total_process_time:.4f}秒",
                    "instance_id": instance.instance_id,
                    "batch_size": processed_sz,
                    "queue_type": instance.name,
                    "audio_duration": f"{req.audio_duration:.2f}秒",
                }
                req.event.set()
                print(f"[DEBUG][{instance.name}] 任务{req.request_id}完成（兜底），总耗时{total_process_time:.3f}秒")

        except Exception as e:
            error_msg = str(e)
            print(f"[ERROR][{instance.name}] 批处理出错：{error_msg}")
            for i, req in enumerate(current_batch[:processed_sz] if processed_sz > 0 else current_batch):
                if req.event.is_set():  
                    continue
                req.result = {
                    "error": error_msg,
                    "status": "error",
                    "batch_size": processed_sz if processed_sz > 0 else len(current_batch),
                    "queue_type": instance.name,
                }
                req.event.set()
        finally:
            # 清理已处理样本的临时文件（提前返回里可能已清）
            for req in current_batch[:processed_sz] if processed_sz > 0 else current_batch:
                try:
                    if os.path.exists(req.temp_path):
                        os.unlink(req.temp_path)
                except Exception as e:
                    print(f"[WARN][{instance.name}] 清理临时文件失败: {e}")

            del results
            gc.collect()

            # 周期性清理显存缓存
            instance.batch_counter += 1
            if instance.batch_counter % EMPTY_CACHE_PERIOD == 0:
                print(f"[DEBUG][{instance.name}] 周期性显存缓存清理（每 {EMPTY_CACHE_PERIOD} 批）")
                torch.cuda.empty_cache()

            with instance.process_lock:
                instance.is_processing = False


def load_models(dual: bool, path):
    """加载模型实例：
       - dual=True：加载短、长两个实例
       - dual=False：只加载一个实例（命名为 'single'，使用短实例的参数）
    """
    global short_instance, long_instance, DUAL_INSTANCES
    DUAL_INSTANCES = dual
    MODEL_PATH=path
    device0 = torch.device("cuda:0")
    device1 = torch.device("cuda:0")  # 如需多卡，可改为 "cuda:1"

    print(f"正在加载模型实例...（dual={int(dual)}）")

    try:
        if dual:
            # 短实例
            short_instance = ModelInstance(
                instance_id=0,
                name="short",
                batch_size=16,
                batch_timeout=0.003,
                max_queue_size=400
            )
            print(f"加载模型到实例 {short_instance.instance_id} ({short_instance.name}) ...")
            short_instance.model = FireRedAsr.from_pretrained("aed", MODEL_PATH, device=device0)

            # 长实例
            long_instance = ModelInstance(
                instance_id=1,
                name="long",
                batch_size=8,
                batch_timeout=0.005,
                max_queue_size=200
            )
            print(f"加载模型到实例 {long_instance.instance_id} ({long_instance.name}) ...")
            long_instance.model = FireRedAsr.from_pretrained("aed", MODEL_PATH, device=device1)

            torch.cuda.empty_cache()
            print(f"[DEBUG] 模型加载完成（双实例），初始GPU显存清理完成")

            short_instance.start()
            long_instance.start()
            print("两个模型实例加载完成！")

        else:
            # 单实例（统一队列，不做分流）
            short_instance = ModelInstance(
                instance_id=0,
                name="single",
                batch_size=16,
                batch_timeout=0.003,
                max_queue_size=600  # 可适当放大
            )
            print(f"加载模型到实例 {short_instance.instance_id} ({short_instance.name}) ...")
            short_instance.model = FireRedAsr.from_pretrained("aed", MODEL_PATH, device=device0)

            long_instance = None  # 明确置空
            torch.cuda.empty_cache()
            print(f"[DEBUG] 模型加载完成（单实例），初始GPU显存清理完成")

            short_instance.start()
            print("单实例模型加载完成！")

    except Exception as e:
        print(f"模型加载失败: {e}")
        traceback.print_exc()
        exit(1)


def total_queue_length():
    # 仅在双实例模式下有意义
    if not DUAL_INSTANCES:
        return len(short_instance.queue) if short_instance else 0
    with global_len_lock:
        slen = len(short_instance.queue) if short_instance else 0
        llen = len(long_instance.queue) if long_instance else 0
        return slen + llen


@app.post("/asr/single", response_model=Dict[str, Any])
async def single_file_asr(
    audio_file: UploadFile = File(...),
    config: Optional[ASRRequest] = None
):
    """异步接口：dual=1 分流到短/长实例；dual=0 统一进入单实例队列"""
    global DUAL_INSTANCES

    # 模型就绪检查
    if DUAL_INSTANCES:
        if not short_instance or not long_instance:
            raise HTTPException(status_code=503, detail="模型未加载")
    else:
        if not short_instance:
            raise HTTPException(status_code=503, detail="模型未加载（单实例）")

    request_id = str(uuid.uuid4())
    config = config or ASRRequest()
    start_time = time.time()
    temp_path = None

    try:
        # 1) 存临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            temp_path = f.name
            f.write(await audio_file.read())

        # 2) 计算时长（单实例模式下仅用于统计展示，不做分流判断）
        audio_duration = get_audio_duration(temp_path)

        # 3) 选择目标实例 & 容量检查
        if DUAL_INSTANCES:
            target = long_instance if audio_duration > SHORT_THRESHOLD else short_instance

            with target.lock:
                current_len = len(target.queue)
            tq_len = total_queue_length()

            if current_len >= target.max_queue_size:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception:
                    pass
                return JSONResponse(
                    content={
                        "status": "error",
                        "error": f"{target.name} 队列拥塞（{current_len}/{target.max_queue_size}），请稍后重试",
                        "queue_type": target.name,
                    },
                    status_code=503,
                )
            if tq_len >= (short_instance.max_queue_size + long_instance.max_queue_size):
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception:
                    pass
                return JSONResponse(
                    content={
                        "status": "error",
                        "error": f"系统繁忙（总队列 {tq_len}），请稍后再试",
                    },
                    status_code=503,
                )
        else:
            # 单实例：不做长/短分流与总队列限制，只检查单实例队列长度
            target = short_instance  # 名字为 "single"
            with target.lock:
                current_len = len(target.queue)
            if current_len >= target.max_queue_size:
                try:
                    if os.path.exists(temp_path):
                        os.unlink(temp_path)
                except Exception:
                    pass
                return JSONResponse(
                    content={
                        "status": "error",
                        "error": f"队列拥塞（{current_len}/{target.max_queue_size}），请稍后重试",
                        "queue_type": target.name,
                    },
                    status_code=503,
                )

        # 4) 入队
        event = asyncio.Event()
        enqueue_time = time.time()
        batch_req = BatchRequest(
            request_id=request_id,
            filename=audio_file.filename,
            temp_path=temp_path,
            config=config.model_dump(),
            result={},
            event=event,
            start_time=start_time,
            enqueue_time=enqueue_time,
            audio_duration=audio_duration,
        )
        with target.cond:
            target.queue.append(batch_req)
            qlen = len(target.queue)
            if DUAL_INSTANCES:
                tq_len = total_queue_length()
                print(f"[DEBUG] 请求{request_id}入{target.name}队列（时长{audio_duration:.2f}秒，队列长度：{qlen}，总队列：{tq_len}）")
            else:
                print(f"[DEBUG] 请求{request_id}入单实例队列（时长{audio_duration:.2f}秒，队列长度：{qlen}）")
            target.cond.notify()

        # 5) 等待结果（双实例：长=180s/短=60s；单实例：统一 90s）
        if DUAL_INSTANCES:
            timeout = 180 if target is long_instance else 60
        else:
            timeout = 90

        try:
            await asyncio.wait_for(event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            with target.cond:
                target.queue = [req for req in target.queue if req.request_id != request_id]
                target.cond.notify_all()
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception:
                pass
            return JSONResponse(
                content={
                    "status": "error",
                    "error": f"请求超时（{timeout}秒）",
                    "queue_type": target.name,
                },
                status_code=504,
            )

        # 6) 返回结果
        if batch_req.result.get("status") == "success":
            return batch_req.result
        else:
            return JSONResponse(content=batch_req.result, status_code=500)

    except Exception as e:
        try:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)
        except Exception:
            pass
        print(f"[ERROR] 请求{request_id}处理出错：{str(e)}")
        return JSONResponse(content={"error": str(e), "status": "error"}, status_code=500)


@app.get("/healthz")
def healthz():
    if DUAL_INSTANCES:
        if not short_instance or not long_instance:
            return JSONResponse(content={"status": "not_ready"}, status_code=503)
        return {
            "status": "ok",
            "short_queue": len(short_instance.queue),
            "long_queue": len(long_instance.queue),
            "mode": "dual"
        }
    else:
        if not short_instance:
            return JSONResponse(content={"status": "not_ready"}, status_code=503)
        return {
            "status": "ok",
            "single_queue": len(short_instance.queue),
            "mode": "single"
        }


if __name__ == '__main__':
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="长短语音分实例ASR服务（提前返回版）")
    parser.add_argument("--dual", type=int, choices=[0, 1],
                        default=int(os.getenv("DUAL_INSTANCES", "1")),
                        help="1=加载两个实例（短/长分流），0=只加载一个实例（统一队列，不做时长分流）")
    parser.add_argument('--model_path', '-m', required=True, help='模型文件路径')
    parser.add_argument('--port',type=int, default=8000, required=True, help='端口号')
    args, _ = parser.parse_known_args()

    # 加载模型
    load_models(dual=args.dual, path=args.model_path)

    # 启动服务
    import uvicorn
    print(f"启动ASR服务（提前返回版，mode={'dual' if DUAL_INSTANCES else 'single'}）...")
    uvicorn.run(
        app,
        host='0.0.0.0',
        port=args.port,
        workers=1,   # 维持单进程：实例在线程内
        reload=False,
        timeout_keep_alive=120
    )
