from ouster.sdk import client

hostname = "192.168.1.10"

source = client.Sensor(hostname)

for scan in source:
    print("Receiving scan")
    break