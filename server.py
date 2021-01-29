import tensorflow as tf
import numpy as np
import miditoolkit
import modules
import pickle
import utils
import time
import argparse
import math

from threading import Thread, Event
from pythonosc import dispatcher
from pythonosc import osc_server
from model import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# TODO: Move to engine.py

state = {
  'isRunning': False,
  'history': [[ 0, 208, 214, 1 ]],
  'temperature': 1.2,
  'tickInterval': 0.25,
  'model': None
}

class Clock(Thread):
    def __init__(self, event):
        Thread.__init__(self)
        self.stopped = event

    def run(self):
        while not self.stopped.wait(state['tickInterval']):
            if (state['isRunning'] == True):
              # print("tick")
              # sample_model(None,[state['model']])
              state['model'].tick()

def start_timer():
  stopFlag = Event()
  state['stopFlag'] = stopFlag
  state['clock'] = Clock(stopFlag)
  state['clock'].start()

def stop_timer():
  # this will stop the timer
  state['stopFlag'].set()

def print_volume_handler(unused_addr, args, volume):
  print("[{0}] ~ {1}".format(args[0], volume))

def print_compute_handler(unused_addr, args, volume):
  try:
    print("[{0}] ~ {1}".format(args[0], args[1](volume)))
  except ValueError: pass

def engine_set(unused_addr, args):
  try:
    field, value = args
    state[field] = value
    print("[{0}] ~ {1}".format(field, value))
  except KeyError:
    print("no such key ~ {0}".format(field))
    pass

def push_event(unused_addr, event):
  print("[event] ~ {0}".format(event))
  state['history'][0].append(event)

def engine_print(unused_addr, args):
  field = args
  try:
    print("[{0}] ~ {1}".format(field, state[field]))
  except KeyError:
    print("no such key ~ {0}".format(field))
    pass

def sample_model(unused_addr, args):
  model = args[0]
  event = model.predict()
  print(event)

def prepare_model(unused_addr, args):
  model = args[0]
  event = model.realtime_setup(state['history'])
  print(event)

def bind_dispatcher(dispatcher, model):
  state['model'] = model
  dispatcher.map("/start", engine_set, 'isRunning', True)
  dispatcher.map("/pause", engine_set, 'isRunning', False)
  dispatcher.map("/reset", lambda _: state['history'].clear())
  dispatcher.map("/debug", engine_print)
  dispatcher.map("/event", push_event) # event2word

  dispatcher.map("/set", lambda addr, k, v: engine_set(addr, [k,v]))

  if (model):
    dispatcher.map("/sample", sample_model, model)
    dispatcher.map("/prep",   prepare_model, model)

def load_model():
  return PopMusicTransformer(checkpoint='REMI-tempo-checkpoint', is_training=False)

# /TODO: Move to engine.py


# Main
if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="127.0.0.1", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=5005, help="The port to listen on")
  args = parser.parse_args()

  model = load_model()
  prepare_model(None,[model]) # for real time use

  dispatcher = dispatcher.Dispatcher()
  bind_dispatcher(dispatcher, model)

  server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))
  start_timer()
  server.serve_forever()
  stop_timer()
  model.close()