var fs = require("fs");
var tf = require("@tensorflow/tfjs");
var tfn = require("@tensorflow/tfjs-node");

var modelPaths = {
  "mobilenet-style": 'saved_model_style_js/model.json',
  "inception-style": 'saved_model_style_inception_js/model.json',
  "original-transformer": 'saved_model_transformer_js/model.json',
  "seperable-transformer": 'saved_model_transformer_separable_js/model.json'
}

var models = {};

async function getModel(name)
{
  //if model isn't loaded yet, load it first
  if(!(name in models))
  {
    var handler = tfn.io.fileSystem(modelPaths[name]);
    var model = await tf.loadGraphModel(handler);
    models[name] = model;
  }

  return models[name];
}

function scaleImage(img, size, keepAspect=true)
{
  var sizeX = size;
  var sizeY = size;
  return tf.tidy(() =>
  {
    //remove alpha channel if present
    img = tf.slice(img, [0, 0, 0], [-1, -1, 3]);

    if(keepAspect)
    {
      //determine new size
      //[height, width, channels]
      var aspect = img.shape[1] / img.shape[0]; //calculate w/h aspect
      if(aspect > 1)
      {
        sizeY /= aspect;
      }
      else
      {
        sizeX *= aspect;
      }
    }

    //resize
    if(size != null)
    {
      img = tf.image.resizeBilinear(img, [sizeY, sizeX]);
    }
    //rescale to 0..1 range
    img = img.toFloat().div(tf.scalar(255));

    return img;
  });
}

function unscaleImage(img)
{
  //scale back to 0..255 range
  return tf.tidy(() =>
  {
    return img.mul(tf.scalar(255));
  });
}

async function loadImage(path, size)
{
  return tf.tidy(() =>
  {
    //TODO: async
    var img = fs.readFileSync(path);
    img = tfn.node.decodeImage(img);
    img = scaleImage(img, size);

    return img;
  });
}

async function saveImage(img, path)
{
  tf.tidy(() =>
  {
    img = unscaleImage(img);
    (async () => {
      img = await tfn.node.encodeJpeg(img);
      fs.writeFileSync(path, img);
    })();
  });
}

async function loadImageB64(b64, size)
{
  return tf.tidy(() =>
  {
    var img = Buffer.from(b64, "base64");
    img = tfn.node.decodeImage(img);
    img = scaleImage(img, size);

    return img;
  });
}

async function saveImageB64(img)
{
  img = unscaleImage(img);
  var disposeMe = img;

  img = await tfn.node.encodeJpeg(img);
  img = Buffer.from(img).toString('base64');
  // img = "data:image/jpeg;base64," + img;

  disposeMe.dispose();
  return img;
}

module.exports = {
  getModel,
  loadImage,
  saveImage,
  loadImageB64,
  saveImageB64
}
