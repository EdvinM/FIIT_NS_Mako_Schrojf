import sys
sys.path.append('../src')

import models.MiniVGGNetBuilder as MiniVGGNetBuilder

print(dir(MiniVGGNetBuilder))
print("\n")

vgg_classes_model = MiniVGGNetBuilder.create_new_model(101)
vgg_regress_model = MiniVGGNetBuilder.create_new_regression_model()

import data.load_data as load_data

print(dir(load_data))
print("\n")

from data.IMDBSequence import IMDBSequence
from data.WIKISequence import WIKISequence

print(dir(IMDBSequence))
print("\n")

from models.MiniVGGNetModel import MiniVGGNetModel

print(dir(MiniVGGNetModel))
print("\n")

import models.VGGFaceModel as VGGFaceModel

print(dir(VGGFaceModel))
print("\n")

print("OK")