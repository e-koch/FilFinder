def get_package_data():
    return {
        _ASTROPY_PACKAGE_NAME_ + '.tests': ['testing_data/*.fits',
                                            'testing_data/test1/*',
                                            'testing_data/test2/*',
                                            'coveragerc']}
