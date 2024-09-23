import struct

def to_int8(value):
  return struct.unpack('b', struct.pack('b', int(value)))[0]

def to_int16(value):
  return struct.unpack('h', struct.pack('h', int(value)))[0]

def to_int32(value):
  return struct.unpack('i', struct.pack('i', int(value)))[0]

def to_int64(value):
  return struct.unpack('q', struct.pack('q', int(value)))[0]

def to_float16(value):
  return struct.unpack('e', struct.pack('e', float(value)))[0]

def to_float32(value):
  return struct.unpack('f', struct.pack('f', float(value)))[0]

def to_float64(value):
  return struct.unpack('d', struct.pack('d', float(value)))[0]