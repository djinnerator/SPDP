# SPDP
Python implementation of SPDP


##### Guide:

###### Compress:

```
from SPDP import spdp

orig_data = # some byte | bytearray | memoryview

# also has a `level` argument that accepts from 0 to 9.
# 0 is fastest rate with lowest compression. 9 is highest and slowest.
# ex. spdp.compress(orig_data, level=3)
comp_data = spdp.compress(orig_data)
````

###### Decompress:

```
from SPDP import spdp

import numpy as np

# data will be a bytearray
data = spdp.decompress(comp_data)
arr = np.frombuffer(data, dtype=orig_data.dtype).reshape(orig_data.shape)
````

Maybe I will include the original dtype and shape with the compressed data so it doesn't need to be passed to the decompress function.





