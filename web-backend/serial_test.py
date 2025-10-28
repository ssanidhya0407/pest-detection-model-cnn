import serial, time

port = 'COM9'
baud = 9600
try:
    s = serial.Serial(port, baud, timeout=1)
except Exception as e:
    print('ERROR opening serial:', e)
    raise SystemExit(1)

print('Opened', port)
# give Arduino time to finish reset
time.sleep(2)

s.reset_input_buffer()

s.write(b'PEST_ON\n')
print('WROTE PEST_ON')

end = time.time() + 4
while time.time() < end:
    line = s.readline()
    if line:
        print('ARDUINO:', line.decode(errors='ignore').strip())
    time.sleep(0.1)

s.write(b'PEST_OFF\n')
print('WROTE PEST_OFF')
end = time.time() + 2
while time.time() < end:
    line = s.readline()
    if line:
        print('ARDUINO:', line.decode(errors='ignore').strip())
    time.sleep(0.1)

s.close()
print('Done')
