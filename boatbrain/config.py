from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Config:
    width: int = 960
    height: int = 720
    cell: int = 4
    road_half_width: float = 56.0
    sensor_angles: tuple = (-55, 0, 55)
    sensor_lengths: tuple = (24, 48, 80, 120, 180)
    sensor_width: float = 6.0
    heading_bins: int = 120
    hidden: int = 4
    dt: float = 1 / 30
    wheelbase: float = 25.0
    servo_degrees: float = 30.0
    acceleration: float = 100.0
    drag: float = 1.1
    max_forward: float = 90.0
    max_reverse: float = 40.0
    car_length: float = 26.0
    car_width: float = 14.0

    def __post_init__(self):
        if self.hidden < 1 or self.cell < 1 or self.heading_bins < 1:
            raise ValueError("hidden, cell and heading_bins must be positive")
        if len(self.sensor_angles) != 3 or len(self.sensor_lengths) != 5:
            raise ValueError("The lab expects three directions and five nested lengths")
        if any(a <= 0 for a in self.sensor_lengths) or sorted(
            self.sensor_lengths
        ) != list(self.sensor_lengths):
            raise ValueError("Sensor lengths must be positive and ascending")
        if self.sensor_width <= 0 or self.wheelbase <= 0 or self.dt <= 0:
            raise ValueError("Sensor width, wheelbase and timestep must be positive")
        if self.width != 960 or self.height != 720:
            raise ValueError("Track coordinates currently require a 960 x 720 world")

    @property
    def inputs(self):
        return len(self.sensor_angles) * len(self.sensor_lengths)

    @property
    def parameters(self):
        return (self.inputs + 1) * self.hidden + (self.hidden + 1) * 2

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data = dict(data)
        for key in ("sensor_angles", "sensor_lengths"):
            if key in data:
                data[key] = tuple(data[key])
        return cls(**data)
