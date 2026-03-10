import math
import random
from dataclasses import dataclass, field
from typing import Optional, List, Literal, Union

import numpy as np
import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.widgets import Slider
import seaborn as sns

sns.set_theme(style="darkgrid")
