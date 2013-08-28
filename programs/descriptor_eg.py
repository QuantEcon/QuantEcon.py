"""
Origin: QEwP by John Stachurski and Thomas J. Sargent
Filename: descriptor_eg.py
Authors: John Stachurski, Thomas J. Sargent
LastModified: 11/08/2013

"""
class Car(object):

    def __init__(self, miles_till_service=1000):

        self.__miles_till_service = miles_till_service
        self.__kms_till_service = miles_till_service * 1.61

    def set_miles(self, value):
        self.__miles_till_service = value
        self.__kms_till_service = value * 1.61

    def set_kms(self, value):
        self.__kms_till_service = value
        self.__miles_till_service = value / 1.61

    def get_miles(self):
        return self.__miles_till_service

    def get_kms(self):
        return self.__kms_till_service

    miles_till_service = property(get_miles, set_miles)
    kms_till_service = property(get_kms, set_kms)
