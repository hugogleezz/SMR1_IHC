from DRCF import *
import powerup.remote as remote

# Start remote API
remote.start_tcp_remote_api(9225, logging=True)
