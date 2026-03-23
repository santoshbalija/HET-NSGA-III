import enum


class Status(enum.Enum):
    INITIALIZED = 1
    IN_QUEUE = 1
    RUNNING = 2
    SUCCESS = 3
    FAILED = 4
    CANCELED = 5


class Job:

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.status = Status.INITIALIZED

        self.start_time = None
        self.exec_time = None
        self.end_time = None

        self.error = None
        self.data = kwargs

    def is_completed(self):
        return self.status in [Status.SUCCESS, Status.FAILED, Status.CANCELED]

    def get(self, key):
        if key in self.__dict__:
            return self.__dict__[key]
        else:
            return self.data.get(key)

    def set(self, key, value):
        if key in self.__dict__:
            self.key = value
        else:
            self.data[key] = value

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        return self.set(key, value)
