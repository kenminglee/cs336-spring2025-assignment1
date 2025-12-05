# Note on encoding-to/decoding-from bytes:

UTF-8/UTF-16/UTF-32 are encoding specs, they determine how strings are encoded in bytes, and vice versa.

1. To encode a `str` into `bytes`, do `<str>.encode('utf-8')`:
```
>>> "hi".encode("utf-8")
b'hi'
>>> "你好".encode("utf-8")
b'\xe4\xbd\xa0\xe5\xa5\xbd'
```

2. To convert `bytes` into `str`, do `<bytes>.decode('utf-8')`:
```
>>> b'hi'.decode('utf-8')
'hi'
```

3. Note that `utf-8` is used by default:

Encoding:
```
>>> 'hi'.encode()
b'hi'
>>> 'hi'.encode('utf-16')
b'\xff\xfeh\x00i\x00'
>>> 'hi'.encode('utf-8')
b'hi'
```

Decoding:
```
>>> b'hi'.decode('utf-8')
'hi'
>>> b'hi'.decode('utf-16')
'楨'
>>> b'hi'.decode()
'hi'
```

4. Iterating a sequence of `bytes` gives us a list of integers (representing the value of each byte in base 10):
```
>>> [i for i in "hi".encode("utf-8")]
[104, 105]
```

5. Hence, to convert a string into a list of bytes, do:
```
>>> [i for i in "hi".encode("utf-8")]
[104, 105]
>>> [bytes([i]) for i in "hi".encode("utf-8")]
[b'h', b'i']
```

Note: `bytes([i])` creates a `bytes` with value `i`, while `bytes(i)` creates `i` bytes with value 0:
```
>>> bytes([2])
b'\x02'
>>> bytes(2)
b'\x00\x00'
```

