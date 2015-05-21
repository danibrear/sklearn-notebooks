import numpy as np
import mahotas as mh
from mahotas.features import surf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import *
from sklearn.cluster import MiniBatchKMeans
import glob

all_instance_filenames = []
all_instance_targets = []

for f in glob.glob('./data/dog-v-cat/train/*.jpg'):
    target = 1 if 'cat' in f else 0
    all_instance_filenames.append(f)
    all_instance_targets.append(target)
surf_features = []
counter = 0
for f in all_instance_filenames:
    print 'Reading image:', f
    image = mh.imread(f, as_grey=True)
    surf_features.append(surf.surf(image)[:, 5:])
