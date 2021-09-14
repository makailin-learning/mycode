from pyinstrument import Profiler
import time
profiler=Profiler()
profiler.start()
#
a=[i for i in range(100000)]
time.sleep(0.1)
#
profiler.stop()
profiler.print()

