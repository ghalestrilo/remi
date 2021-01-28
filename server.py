import tensorflow as tf
import numpy as np
import miditoolkit
import modules
import pickle
import utils
import time

from model import PopMusicTransformer
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

"""Small example OSC server

This program listens to several addresses, and prints some information about
received packets.
"""
import argparse
import math

from pythonosc import dispatcher
from pythonosc import osc_server

# TODO: Move to engine.py

state = {
  'isRunning': False,
  'history': [],
  'temperature': 1.2
}

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
  state['history'].append(event)

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

def bind_dispatcher(dispatcher, model):
  dispatcher.map("/filter", print)
  dispatcher.map("/volume", print_volume_handler, "Volume")
  dispatcher.map("/logvolume", print_compute_handler, "Log volume", math.log)

  dispatcher.map("/start", engine_set, 'isRunning', True)
  dispatcher.map("/pause", engine_set, 'isRunning', False)
  dispatcher.map("/reset", lambda: state['history'].clear())

  if (model):
    dispatcher.map("/sample", sample_model, model)
  dispatcher.map("/debug", engine_print)
  dispatcher.map("/event", push_event) # event2word

def load_model():
  return PopMusicTransformer(checkpoint='REMI-tempo-checkpoint', is_training=False)

# /TODO: Move to engine.py



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="127.0.0.1", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=5005, help="The port to listen on")
  args = parser.parse_args()

  model = load_model()

  dispatcher = dispatcher.Dispatcher()
  bind_dispatcher(dispatcher, model)

  server = osc_server.ThreadingOSCUDPServer((args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()

  model.close()