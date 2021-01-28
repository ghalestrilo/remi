import tensorflow as tf
import numpy as np
import miditoolkit
import modules
import pickle
import utils
import time


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
  'history': []
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

def bind_dispatcher(dispatcher):
  dispatcher.map("/filter", print)
  dispatcher.map("/volume", print_volume_handler, "Volume")
  dispatcher.map("/logvolume", print_compute_handler, "Log volume", math.log)

  dispatcher.map("/start", engine_set, 'isRunning', True)
  dispatcher.map("/pause", engine_set, 'isRunning', False)
  dispatcher.map("/reset", engine_set, 'history',   list([]))

  dispatcher.map("/debug", engine_print)
  dispatcher.map("/event", push_event)
# /TODO: Move to engine.py



if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--ip",
      default="127.0.0.1", help="The ip to listen on")
  parser.add_argument("--port",
      type=int, default=5005, help="The port to listen on")
  args = parser.parse_args()

  dispatcher = dispatcher.Dispatcher()
  bind_dispatcher(dispatcher)

  server = osc_server.ThreadingOSCUDPServer(
      (args.ip, args.port), dispatcher)
  print("Serving on {}".format(server.server_address))
  server.serve_forever()