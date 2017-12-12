# ADFIS: Anti-"Donut Fairy" Intrusion System.

```
   _____  ________  ___________.___  _________
  /  _  \ \______ \ \_   _____/|   |/   _____/
 /  /_\  \ |    |  \ |    __)  |   |\_____  \
/    |    \|    `   \|     \   |   |/        \
\____|__  /_______  /\___  /   |___/_______  /
        \/        \/     \/                \/

May the "Donut Fairy" NOT be with you!
```

## Based on

- https://github.com/Hironsan/BossSensor
- https://github.com/webb04/NetflixPauser

## Requirements

* WebCamera
* Python2
* Debian (PRs to support other OS are welcome too).
* Lots of images of you and other persons.

Put your images into `data/user` (named `#_user.jpg`, where `#` is an incremental number) and the other images into `data/others`.

## Installation

```bash
git clone https://github.com/Ismael-VC/ADFIS
cd ADFIS
chmod +x install.sh anti-donut_fairy.py user_train.py
./install.sh
```

## Usage
First, train user image.

```bash
$ ./user_train.py
```

Second, start the *Anti-"Donut Fairy" Intrusion System*.

```bash
$ adfis
```

Third, keep doing your job and it will lock your screen when you either
walk away, turn your head around or cover the webcam.

## Notes

- :warning: Warning:
    - **Don't be naive, this is a prototype, and even then, never forget that the
      all seing eye of the "Donut Fairy" will always be lurking around the corner,
      ...waiting for our logic to fail misserably!**
    - Expect false positives and negatives, until thorough testing.
- Please report back any bug in the issues section or better yet, send a PR.

## To do

- Support for other operating systems.
- Tweak parameters sensibility and timing defaults.
- Test model accuracy with larger number of images.
- Add support for other backends (Theano).

