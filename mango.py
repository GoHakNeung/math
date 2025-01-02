with open('/content/math/mango.txt', 'r') as file:
    berry = file.read().strip()
b = 'sk-proj-' + berry[:17] + berry[22:68] + berry[73:116] + berry[121:159] + berry[164:]
print(b, end = '')
