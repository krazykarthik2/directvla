class Message:
    def __init__(self, b=None):
        if b is None:
            self.header = bytes([0xAA, 0xAA])
            self.len = 2
            self.id = 0
            self.ctrl = 0
            self.params = bytearray()
            self.checksum = None
        else:
            self.header = b[0:2]
            self.len = b[2]
            self.id = b[3]
            self.ctrl = b[4]
            self.params = bytearray(b[5:-1])
            self.checksum = b[-1]

    def __repr__(self):
        return f"Message(id={self.id}, ctrl={self.ctrl}, params={list(self.params)})"

    def refresh(self):
        if isinstance(self.params, (bytes, bytearray)):
            self.len = 2 + len(self.params)
        else:
            self.len = 2
        checksum = self.id + self.ctrl
        for b in self.params:
            checksum += b
        checksum = (256 - (checksum % 256)) % 256
        self.checksum = checksum

    def bytes(self):
        self.refresh()
        cmd = bytearray([0xAA, 0xAA, self.len, self.id, self.ctrl])
        cmd.extend(self.params)
        cmd.append(self.checksum)
        return bytes(cmd)
