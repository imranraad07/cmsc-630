from math import ceil
from multiprocessing import Manager
from threading import Thread
from typing import List, Dict, Union, Callable
from typing import Union, List
from multiprocessing import Queue
from tqdm.auto import tqdm
from joblib import Parallel, delayed


# https://towardsdatascience.com/parallel-batch-processing-in-python-8dcce607d226

def batch_process(items: list, function: Callable, batch_size: int = 10, *args, **kwargs, ) -> \
  List[Dict[str, Union[str, List[str]]]]:
  n_workers = int(len(items) / batch_size)
  batches = [
    items[ix:ix + batch_size]
    for ix in range(0, len(items), batch_size)
  ]

  totals = len(items)
  print('totals', totals)
  manager = Manager()
  queue = manager.Queue()
  try:
    progproc = Thread(target=progress_bar, args=(totals, queue))
    progproc.start()
    result = Parallel(n_jobs=n_workers)(
      delayed(task_wrapper)(pid, function, batch, queue, *args, **kwargs) for pid, batch in enumerate(batches))
  finally:
    queue.put('done')
    progproc.join()

  flattened = [item for sublist in result for item in sublist]

  return flattened


def task_wrapper(pid, function, batch, queue, *args, **kwargs):
  result = []
  for example in batch:
    result.append(function(example, *args, **kwargs))
    queue.put(f'update {pid}')
  return result


def progress_bar(totals: Union[int, List[int]], queue: Queue, ) -> None:
  if isinstance(totals, list):
    splitted = True
    pbars = [
      tqdm(desc=f'Worker {pid + 1}', total=total, position=pid, )
      for pid, total in enumerate(totals)
    ]
  else:
    splitted = False
    pbars = [
      tqdm(total=totals)
    ]

  while True:
    try:
      message = queue.get()
      print(message)
      if message.startswith('update'):
        if splitted:
          pid = int(message[6:])
          pbars[pid].update(1)
        else:
          pbars[0].update(1)
      elif message == 'done':
        break
    except:
      pass
  for pbar in pbars:
    pbar.close()