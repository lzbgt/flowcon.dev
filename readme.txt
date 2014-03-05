# collector
#  zmq version
python -m flowcon.main -i "tcp://10.1.31.81:5556" -q "tcp://*:5567" -s "tcp://*:5568"
#  native version
python -m flowcon.main -i "udp://10.215.1.6:2059" -q "tcp://*:5567" -s "tcp://*:5568"
python -m flowcon.main -i "udp://10.1.32.196:2059" -q "tcp://*:5569" -s "tcp://*:5570"

# tester
python -m test.query -i "tcp://localhost:5567" -f sourceIPv4Address destinationIPv4Address -p 5

# rewrite
## macs first
tcprewrite --dlt=enet -C --enet-dmac=50:e5:49:64:67:44 --enet-smac=50:e5:49:64:67:44 --infile=/media/store/workspace/calix/captures/udp2059.pcap --outfile=/media/store/workspace/calix/captures/re2059.pcap
## then IPs
tcprewrite -C --dstipmap=10.215.1.6/32:192.168.1.82/32 --srcipmap=0.0.0.0/0:192.168.1.82/32 --infile=/media/store/workspace/calix/captures/re2059.pcap --outfile=/media/store/workspace/calix/captures/u2059.pcap

# or
tcprewrite -C --dstipmap=10.1.31.81/32:54.200.102.219/32 --srcipmap=0.0.0.0/0:192.168.1.65/32 --enet-dmac=ac:5d:10:33:6b:39 --enet-smac=00:e5:49:64:67:44 --infile=/media/store/workspace/calix/captures/stream/raw.pcap --outfile=/media/store/workspace/calix/captures/stream/reraw.pcap

# send
sudo tcpreplay --intf1=eth1 /media/store/workspace/calix/captures/stream/reraw.pcap
