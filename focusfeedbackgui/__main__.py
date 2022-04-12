#!/usr/bin/env python

from argparse import ArgumentParser
from focusfeedbackgui import app
from focusfeedbackgui.cylinderlens import calibrate_z
from focusfeedbackgui.utilities import warp, info


def main():
    parser = ArgumentParser(description='Tracking using a cylindrical lens.')
    parser.add_argument('mode', help='track|calibrate|transform|info', type=str, default='track', nargs='?')
    parser.add_argument('file', help='image_file', type=str, default=None, nargs='?')
    parser.add_argument('out', help='path to tif out', type=str, default=None, nargs='?')
    parser.add_argument('-c', '--channel', help='channel', type=int, default=None)
    parser.add_argument('-z', '--zslice', help='z-slice', type=int, default=None)
    parser.add_argument('-t', '--time', help='time', type=int, default=None)
    parser.add_argument('-s', '--split', help='split channels', action='store_true')
    parser.add_argument('-f', '--force', help='force overwrite', action='store_true')
    parser.add_argument('-e', '--emission', help='emission wavelength', type=float, default=None)
    args = parser.parse_args()

    if args.mode.lower().startswith('track'):
        app.main()
    elif args.mode.lower().startswith('cali'):
        calibrate_z(args.file, args.emission, args.channel)
    elif args.mode.lower().startswith('trans') or args.mode.lower().startswith('warp'):
        warp(args.file, args.out, args.channel, args.zslice, args.time, args.split, args.force)
    elif args.mode.lower().startswith('info'):
        info(args.file)


if __name__ == '__main__':
    main()
