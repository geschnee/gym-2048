
basically this repo:

https://github.com/activatedgeek/gym-2048


remove the dumb requirements.txt
(requires old versions with the ~= specifier)

remove the requirements.txt

the install like this:

python setup.py install

<--- do this in admin mode

https://pypi.org/project/gym-2048/


Salina gym.py has Problems with the np.rot90 Method.

This error is raised:
ValueError: At least one stride in the given numpy array is negative, and tensors with negative strides are not currently supported. (You can probably work around this by making a copy of your array  with array.copy().)

To prevent this error, the numpy array (observation space) has to be copied ("forgets the negative indices").

See .copy() in env.py at return of step


upload to pypi using twine works like this:

twine upload -r pypi dist/*