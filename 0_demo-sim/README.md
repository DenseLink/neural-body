***INSERT GRAPHIC HERE (include hyperlink in image)***

# Neural Body N-Body Simulator
> By The AstroGators

An n-body simulator powered by a neural network.

Neural Body is an n-body simulator currently as an alpha demonstration of substituting 
calculations for planetary motion with a neural network.  Currently, the neural 
networks can predict the position of Mars or Pluto given the rest of the positions
in the planetary system.  

The eventual goal is to replace the physics simulator that calculates the positions
of other bodies to feed the neural network completely with neural networks that 
take in an acceleration vector on each body along with a desired simulation time 
step and perform the integration necessary to calculate the displacement of the body.

[![Game Overview Image](https://raw.githubusercontent.com/nedgar76/neural-body/demo-sim/0_demo-sim/readme_resources/overview_screenshot.png?token=ALC2NMM5G56RZFD237TQX32677FSA)]()


- Most people will glance at your `README`, *maybe* star it, and leave
- Ergo, people should understand instantly what your project is about based on your repo

> Tips

- HAVE WHITE SPACE
- MAKE IT PRETTY
- GIFS ARE REALLY COOL

> GIF Tools

- Use <a href="http://recordit.co/" target="_blank">**Recordit**</a> to create quicks screencasts of your desktop and export them as `GIF`s.
- For terminal sessions, there's <a href="https://github.com/chjj/ttystudio" target="_blank">**ttystudio**</a> which also supports exporting `GIF`s.

**Recordit**

![Recordit GIF](http://g.recordit.co/iLN6A0vSD8.gif)

**ttystudio**

![ttystudio GIF](https://raw.githubusercontent.com/chjj/ttystudio/master/img/example.gif)

---

## Table of Contents (Optional)

> If your `README` has a lot of info, section headers might be nice.

- [Requirements](#requirements)
- [Installation](#installation)
- [Features](#features)
- [Team](#team)
- [FAQ](#faq)
- [License](#license)


---

## Example (Optional)

```javascript
// code away!

let generateProject = project => {
  let code = [];
  for (let js = 0; js < project.length; js++) {
    code.push(js);
  }
};
```

---
## Installation
### Requirements
- Compatible debian-based Linux distro.  Ubuntu 20.04 Preferred.
- Python 3.8 or higher.
- PyGame 2.0.0.dev10 or higher.
- TensorFlow 2.2.0
- Pandas 1.0.5
- Numpy 1.19.0
All dependencies above except for Python 3.8 should install when `pip install` is run.

### Setup
- Download `neural_body-0.1.0.tar.gz` from the `dist` directory.
- Navigate to local folder where download is located.
- Use `pip install neural_body`
- Use the `neural_body` command to run the simulator.

[![Game Overview Image](https://raw.githubusercontent.com/nedgar76/neural-body/demo-sim/0_demo-sim/readme_resources/overview_screenshot.png?token=ALC2NMM5G56RZFD237TQX32677FSA)]()

---

## Usage
## Features
## Documentation
Documentation compiled with Sphinx is included in the 
## Tests (Optional)

- Going into more detail on code and technologies used
- I utilized this nifty <a href="https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet" target="_blank">Markdown Cheatsheet</a> for this sample `README`.

---
## Team
The AstroGators formed as a result of the "CIS4930 - Performant Python Programming" 
course at the University of Florida.

Team members include:
- Nathaniel Edgar
- Craig Boger
- Gary Jones
- Cory Robertson
- Andrew Sowinski
 
The Github repo for this initial demonstration is located at: 
<a href="https://github.com/nedgar76/neural-body/tree/demo-sim/0_demo-sim" target="_blank"> https://github.com/nedgar76/neural-body/tree/demo-sim/0_demo-sim </a>

> Or Contributors/People

| <a href="http://fvcproductions.com" target="_blank">**FVCproductions**</a> | <a href="http://fvcproductions.com" target="_blank">**FVCproductions**</a> | <a href="http://fvcproductions.com" target="_blank">**FVCproductions**</a> |
| :---: |:---:| :---:|
| [![FVCproductions](https://avatars1.githubusercontent.com/u/4284691?v=3&s=200)](http://fvcproductions.com)    | [![FVCproductions](https://avatars1.githubusercontent.com/u/4284691?v=3&s=200)](http://fvcproductions.com) | [![FVCproductions](https://avatars1.githubusercontent.com/u/4284691?v=3&s=200)](http://fvcproductions.com)  |
| <a href="http://github.com/fvcproductions" target="_blank">`github.com/fvcproductions`</a> | <a href="http://github.com/fvcproductions" target="_blank">`github.com/fvcproductions`</a> | <a href="http://github.com/fvcproductions" target="_blank">`github.com/fvcproductions`</a> |

- You can just grab their GitHub profile image URL
- You should probably resize their picture using `?s=200` at the end of the image URL.

---

## FAQ

- **How do I do *specifically* so and so?**
    - No problem! Just do this.

---

## Donations (Optional)

- You could include a <a href="https://cdn.rawgit.com/gratipay/gratipay-badge/2.3.0/dist/gratipay.png" target="_blank">Gratipay</a> link as well.

[![Support via Gratipay](https://cdn.rawgit.com/gratipay/gratipay-badge/2.3.0/dist/gratipay.png)](https://gratipay.com/fvcproductions/)


---

## License

[![License](http://img.shields.io/:license-mit-blue.svg?style=flat-square)](http://badges.mit-license.org)

- **[MIT license](http://opensource.org/licenses/mit-license.php)**
- Copyright 2015 © <a href="http://fvcproductions.com" target="_blank">FVCproductions</a>.