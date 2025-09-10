from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
def multi_thread_task_dict(task_dictionary, num_workers=1, show_progress=True):
    final_results = {}
    futures = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for id_, task in task_dictionary.items():
            futures.append(
                executor.submit(
                    lambda id_=id_, task=task: {"id": id_, "task_result": task()}
                )
            )
        if show_progress:
            with tqdm(total=len(futures)) as pbar:
                for future in as_completed(futures):
                    result = future.result()
                    final_results[result["id"]] = result["task_result"]
                    pbar.update(1)
        else:
            for future in as_completed(futures):
                result = future.result()
                final_results[result["id"]] = result["task_result"]
    return final_results