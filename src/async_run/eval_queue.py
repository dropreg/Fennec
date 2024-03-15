from utils import common_utils
import threading
import queue
import time
from collections import defaultdict
from data.dialogue import Dialogue


class AsyncEvalQueue:
    def __init__(self, concurrent):
        self.logger = common_utils.get_loguru()

        self.eval_queue = queue.Queue()
        self.lock = threading.Lock()
        self.thread_list = []

        self.max_retries = defaultdict(str)
        self.concurrent = concurrent
        self.max_retry = 3

    def push(self, data):
        self.eval_queue.put(data)

    def run(self, process_func, pipe):
        self.logger.info(
            "队列调度开始 总长度 {} 并发数量 {}".format(self.eval_queue.qsize(), self.concurrent)
        )

        def async_func(thread_idx, async_process_func, pipe):
            while not self.eval_queue.empty():
                start_time = time.time()
                self.logger.info("队列{}->处理数据开始".format(thread_idx))

                if self.eval_queue.empty():
                    break

                data_idx, data = self.eval_queue.get()

                result = None
                if False:
                    result = async_process_func(pipe, data_idx, data, thread_idx)
                else:
                    try:
                        result = async_process_func(pipe, data_idx, data, thread_idx)
                    except Exception as e:
                        self.logger.info(
                            "队列{}->处理出错 异常信息：{} 重新加入队列".format(thread_idx, e)
                        )

                        if isinstance(data, Dialogue):
                            session_id = data.get_session_id()
                            if session_id not in self.max_retries:
                                self.max_retries[session_id] = 0
                            else:
                                self.max_retries[session_id] += 1

                            if self.max_retries[session_id] < self.max_retry:
                                self.eval_queue.put(data)
                            else:
                                self.logger.info(
                                    "队列{}->处理出错 超过{}次重试".format(
                                        thread_idx, self.max_retry
                                    )
                                )
                        elif isinstance(data, list) and isinstance(data[0], Dialogue):
                            for data_item in data:
                                session_id = data_item.get_session_id()
                                if session_id not in self.max_retries:
                                    self.max_retries[session_id] = 0
                                else:
                                    self.max_retries[session_id] += 1

                                if self.max_retries[session_id] < self.max_retry:
                                    self.eval_queue.put(data)
                                else:
                                    self.logger.info(
                                        "队列{}->处理出错 超过{}次重试".format(
                                            thread_idx, self.max_retry
                                        )
                                    )
                        if result is None:
                            continue

                self.logger.info(
                    "队列{}->处理数据完成, 耗时 {} s 剩余 {}".format(
                        thread_idx,
                        round(time.time() - start_time, 3),
                        self.eval_queue.qsize(),
                    )
                )

        for thread_idx in range(self.concurrent):
            consumer_thread = threading.Thread(
                target=async_func, args=(thread_idx, process_func, pipe)
            )
            consumer_thread.start()
            self.thread_list.append(consumer_thread)

        for consumer_thread in self.thread_list:
            consumer_thread.join()

        self.logger.info("队列调度结束，总长度 {}".format(self.eval_queue.qsize()))
