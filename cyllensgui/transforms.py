import SimpleITK as sitk  # best if SimpleELastix is installed: https://simpleelastix.readthedocs.io/GettingStarted.html
import yaml
import os
import numpy as np
from dill import register
from copy import deepcopy

if __package__ == '':
    from utilities import yamlload
else:
    from .utilities import yamlload


class Transform:
    def __init__(self, *args):
        self.transform = sitk.ReadTransform(os.path.join(os.path.dirname(__file__), 'transform.txt'))
        if len(args) == 1:  # load from file or dict
            self.load(*args)
        elif len(args) == 2:  # make new transform using fixed and moving image
            self.register(*args)
        self._last = None

    def copy(self):
        return deepcopy(self)

    @staticmethod
    def _get_matrix(value):
        return np.array(((*value[:2], value[4]), (*value[2:4], value[5]), (0, 0, 1)))

    @property
    def matrix(self):
        return self._get_matrix(self.parameters)

    @matrix.setter
    def matrix(self, value):
        value = np.asarray(value)
        self.parameters = [*value[0, :2], *value[1, :2], *value[:2, 2]]

    @property
    def dmatrix(self):
        return self._get_matrix(self.dparameters)

    @dmatrix.setter
    def dmatrix(self, value):
        value = np.asarray(value)
        self.dparameters = [*value[0, :2], *value[1, :2], *value[:2, 2]]

    @property
    def parameters(self):
        return self.transform.GetParameters()

    @parameters.setter
    def parameters(self, value):
        value = np.asarray(value)
        self.transform.SetParameters(value.tolist())

    @property
    def origin(self):
        return self.transform.GetFixedParameters()

    @origin.setter
    def origin(self, value):
        value = np.asarray(value)
        self.transform.SetFixedParameters(value.tolist())

    @property
    def inverse(self):
        if self._last is None or self._last != self.asdict():
            self._last = self.asdict()
            self._inverse = Transform(self.asdict())
            self._inverse.transform = self._inverse.transform.GetInverse()
        return self._inverse

    def adapt(self, origin, shape):
        self.origin -= np.array(origin) + (self.shape - np.array(shape)[:2]) / 2
        self.shape = shape[:2]

    def asdict(self):
        return {'CenterOfRotationPoint': self.origin, 'Size': self.shape,
                'TransformParameters': self.parameters, 'dTransformParameters': self.dparameters}

    def frame(self, im, default=0):
        dtype = im.dtype
        im = im.astype('float')
        intp = sitk.sitkBSplineResamplerOrder3 if np.issubdtype(dtype, np.floating) else sitk.sitkNearestNeighbor
        return self.castArray(sitk.Resample(self.castImage(im), self.transform, intp, default)).astype(dtype)

    def coords(self, array):
        return np.array([self.transform.TransformPoint(i.tolist()) for i in np.asarray(array)])

    def save(self, file):
        """ save the parameters of the transform calculated
            with affine_registration to a yaml file
        """
        if not file[-3:] == 'yml':
            file += '.yml'
        with open(file, 'w') as f:
            yaml.safe_dump(self.asdict(), f, default_flow_style=None)

    def load(self, file):
        """ load the parameters of a transform from a yaml file or a dict
        """
        if isinstance(file, dict):
            d = file
        else:
            if not file[-3:] == 'yml':
                file += '.yml'
            with open(file, 'r') as f:
                d = yamlload(f)
        self.origin = [float(i) for i in d['CenterOfRotationPoint']]
        self.parameters = [float(i) for i in d['TransformParameters']]
        self.dparameters = [float(i) for i in d['dTransformParameters']] \
            if 'dTransformParameters' in d else 6 * [np.nan]
        self.shape = [float(i) for i in d['Size']]

    def register(self, fix, mov):
        self.shape = fix.shape
        fix, mov = self.castImage(fix), self.castImage(mov)
        if hasattr(sitk, 'ElastixImageFilter'):
            tfilter = sitk.ElastixImageFilter()
            tfilter.LogToConsoleOff()
            tfilter.SetFixedImage(fix)
            tfilter.SetMovingImage(mov)
            tfilter.SetParameterMap(sitk.GetDefaultParameterMap('affine'))
            tfilter.Execute()
            transform = tfilter.GetTransformParameterMap()[0]
            self.parameters = [float(t) for t in transform['TransformParameters']]
            self.origin = [float(t) for t in transform['CenterOfRotationPoint']]
            self.shape = [float(t) for t in transform['Size']]
        else:
            # TODO: make this as good as possible, because installing SimpleElastix is difficult
            print('SimpleElastix is not installed, trying SimpleITK, which does not give very accurate results')
            initial_transform = sitk.CenteredTransformInitializer(fix, mov, sitk.AffineTransform(2),
                                                                  sitk.CenteredTransformInitializerFilter.GEOMETRY)
            reg = sitk.ImageRegistrationMethod()
            reg.SetMetricAsMattesMutualInformation(numberOfHistogramBins=512)
            reg.SetMetricSamplingStrategy(reg.RANDOM)
            reg.SetMetricSamplingPercentage(1)
            reg.SetInterpolator(sitk.sitkLinear)
            reg.SetOptimizerAsGradientDescent(learningRate=1.0, numberOfIterations=1000,
                                estimateLearningRate=reg.Once, convergenceMinimumValue=1e-12, convergenceWindowSize=10)
            reg.SetOptimizerScalesFromPhysicalShift()
            reg.SetInitialTransform(initial_transform, inPlace=False)
            reg.SetShrinkFactorsPerLevel(shrinkFactors=[4, 2, 1])
            reg.SetSmoothingSigmasPerLevel(smoothingSigmas=[2, 1, 0])
            reg.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()
            self.transform = reg.Execute(fix, mov)
        self.dparameters = 6 * [np.nan]

    @staticmethod
    def castImage(im):
        if not isinstance(im, sitk.Image):
            im = sitk.GetImageFromArray(im)
        return im

    @staticmethod
    def castArray(im):
        if isinstance(im, sitk.Image):
            im = sitk.GetArrayFromImage(im)
        return im


@register(Transform)
def dill_transform(pickler, obj):
    pickler.save_reduce(lambda d: Transform(d), obj.asdict(), obj=obj)
