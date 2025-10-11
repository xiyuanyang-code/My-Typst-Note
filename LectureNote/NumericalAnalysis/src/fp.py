# https://stackoverflow.com/a/16444778
import struct

def fp64_binary(num):
    return "".join("{:0>8b}".format(c) for c in struct.pack("!d", num))


print(fp64_binary(0.1))
