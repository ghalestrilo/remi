from model import PopMusicTransformer
import numpy as np

class OrnetteModule(PopMusicTransformer):
    def __init__(self, state={}, checkpoint='REMI-tempo-checkpoint'):
      super().__init__(checkpoint, is_training=False)
      self.realtime_ready = False
      self.temperature=1.2
      self.server_state = state
      self.words = []

    def realtime_setup(self, state):
      """Initializes internal model state to be used in real-time

        Parameters:
        state (dict): Ornette Server state, passed via server

        Returns: Nothing
      """
      self.tempo_classes = [v for k, v in self.event2word.items() if 'Tempo Class' in k]
      self.tempo_values = [v for k, v in self.event2word.items() if 'Tempo Value' in k]
      self.batch_m = [np.zeros((self.mem_len, self.batch_size, self.d_model), dtype=np.float32) for _ in range(self.n_layer)]

      self.words = state['history']
      if (self.words == None):
        self.words = []
        ws = [self.event2word['Bar_None']]
        ws.append(np.random.choice(self.tempo_classes))
        ws.append(np.random.choice(self.tempo_values))
        ws.append(self.event2word['Position_1/16'])
        self.words.append(ws)

      self.update_feed_dict()
      self.realtime_ready = True

    def predict(self, temperature=1.2, topk=1):
      if (self.realtime_ready != True):
        return

      _logits, _new_mem = self.sess.run([self.logits, self.new_mem], feed_dict=self.feed_dict)
      _logit = _logits[-1, 0]

      self.batch_m = _new_mem

      word = self.temperature_sampling(
        logits=_logit,
        temperature=temperature,
        topk=topk)

      self.words[0].append(word)
      self.update_feed_dict()
      return word

    def tick(self, topk=1):
      """ Generates the next n tokens
      
      Parameters: None

      Returns: The updated token history
      """
      ev = ''
      while (ev.startswith('Position') == False):
        word = self.predict(self.temperature,topk)
        ev = self.word2event[word]
      return self.words[0] + [word]

    def decode(self, token):
      event_name, event_value = self.word2event.get(token).split('_')
      return (event_name, event_value)

    def update_feed_dict(self):
      if (self.realtime_ready != True):
        original_length = len(self.words[0])
        temp_x = np.zeros((self.batch_size, original_length))
        for b in range(self.batch_size):
            for z, t in enumerate(self.words[b]):
                temp_x[b][z] = t
      else:
        temp_x = np.zeros((self.batch_size, 1))
        for b in range(self.batch_size):
          temp_x[b][0] = self.words[b][-1]

      self.feed_dict = { self.x: temp_x }
      for m, m_np in zip(self.mems_i, self.batch_m):
          self.feed_dict[m] = m_np


## TODO: Module treats own behavior with (Note On, Position, Etc...)