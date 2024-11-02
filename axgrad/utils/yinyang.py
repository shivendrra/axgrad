"""
  @utils/yinyang.py data generation codes
  @brief generates yinyang dateset randomly, from: https://github.com/lkriener/yin_yang_data_set
  comments:
  - also contains the Random Number Generator class (RNG)
  - using xorshift: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
"""

class RNG:
  def __init__(self, seed):
    self.state = seed

  def random_u32(self):
    # doing & 0xFFFFFFFFFFFFFFFF is the same as cast to uint64 in C
    # doing & 0xFFFFFFFF is the same as cast to uint32 in C
    self.state ^= (self.state >> 12) & 0xFFFFFFFFFFFFFFFF
    self.state ^= (self.state << 25) & 0xFFFFFFFFFFFFFFFF
    self.state ^= (self.state >> 27) & 0xFFFFFFFFFFFFFFFF
    return ((self.state * 0x2545F4914F6CDD1D) >> 32) & 0xFFFFFFFF

  def random(self):
    # random float32 in [0, 1)
    return (self.random_u32() >> 8) / 16777216.0

  def uniform(self, a=0.0, b=1.0):
    # random float32 in [a, b)
    return a + (b-a) * self.random()

## dataset ------------------------------------

def yingyang_dataset(random:RNG, n=1000, r_small=0.1, r_big=0.5):
  pts = []
  def dist_to_right_dot(x, y): return ((x - 1.5 * r_big)**2 + (y - r_big)**2)**0.5
  def dist_to_left_dot(x, y): return ((x - 0.5 * r_big)**2 + (y - r_big)**2)**0.5

  def which_class(x, y):
    d_right = dist_to_right_dot(x, y)
    d_left = dist_to_left_dot(x, y)

    criterion1 = d_right <= r_small
    criterion2 = d_left > r_small and d_left <= 0.5 * r_big
    criterion3 = y > r_big and d_right > 0.5 * r_big
    is_yin = criterion1 or criterion2 or criterion3
    is_circles = d_right < r_small or d_left < r_small

    if is_circles:
      return 2
    return 0 if is_yin else 1
  
  def get_sample(goal_class=None):
    while True:
      x = random.uniform(0, 2 * r_big)
      y = random.uniform(0, 2 * r_big)
      if ((x - r_big)**2 + (y - r_big)**2) ** 0.5 > r_big:
        continue
      c = which_class(x, y)
      if goal_class is None or c == goal_class:
        scaled_x = (x / r_big - 1) * 2
        scaled_y = (y / r_big - 1) * 2
        return [scaled_x, scaled_y, c]
    
  for i in range(n):
    goal_class = i % 3
    x, y, c = get_sample(goal_class)
    pts.append([[x, y], c])

  tr = pts[:int(0.8 * n)]
  val = pts[int(0.8 * n):int(0.9 * n)]
  te = pts[int(0.9 * n):]
  return tr, val, te  