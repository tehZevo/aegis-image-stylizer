require("@tensorflow/tfjs-node");
var tf = require("@tensorflow/tfjs");
var ProtoPost = require("protopost");
var utils = require("./utils");

var PORT = parseInt(process.env["PORT"] || 80);
var STYLE_NAME = process.env["STYLE_MODEL"] || "mobilenet-style";
var TRANSFORMER_NAME = process.env["TRANSFORMER_MODEL"] || "seperable-transformer";
var SIZE = parseInt(process.env["OUTPUT_SIZE"] || 512);

var STYLE_SCALE = parseFloat(process.env["STYLE_SCALE"] || 1/2);
var STYLE_RATIO = parseFloat(process.env["STYLE_RATIO"] || 0.8); //0..1

var STYLE_SIZE = Math.floor(SIZE * STYLE_SCALE);

//given an image tensor, returns a 100d style vector
async function getStyle(styleImg)
{
  var styleNet = await utils.getModel(STYLE_NAME);

  return await tf.tidy(() => {
    return styleNet.predict(styleImg.expandDims());
  });
}

async function combineStyles(styles, weights)
{
  return tf.tidy(() =>
  {
    weights = weights || styles.map((e) => 1);

    var combinedStyle = tf.zerosLike(styles[0]);
    for(var i = 0; i < styles.length; i++)
    {
      //scale each style by its weight and add to combined
      var scaledStyle = styles[i].mul(tf.scalar(weights[i]));
      combinedStyle = combinedStyle.add(scaledStyle);
    }

    //average
    combinedStyle = combinedStyle.div(tf.sum(weights));

    return combinedStyle;
  });
}

async function stylize(image, style)
{
  //get model
  var transformerNet = await utils.getModel(TRANSFORMER_NAME);

  return await tf.tidy(() =>
  {
    return transformerNet.predict([image.expandDims(), style]).squeeze();
  });
}

async function stylizeB64(data)
{
  var content = data.content;
  var styles = data.styles;

  console.log("received content image and " + styles.length + " style images");

  content = await utils.loadImageB64(content, SIZE);
  styles = await Promise.all(styles.map((e) => utils.loadImageB64(e, STYLE_SIZE)));
  var disposeUs = styles;
  styles = await Promise.all(styles.map((e) => getStyle(e)));
  disposeUs.forEach((e) => e.dispose());

  var sourceStyle = await getStyle(content);
  var targetStyle = await combineStyles(styles);
  styles.forEach((e) => e.dispose());

  var style = await combineStyles([sourceStyle, targetStyle], [1-STYLE_RATIO, STYLE_RATIO]);
  sourceStyle.dispose();
  targetStyle.dispose();

  var stylized = await stylize(content, style);
  var disposeMe = stylized;
  stylized = await utils.saveImageB64(stylized);
  disposeMe.dispose();
  content.dispose();
  style.dispose();

  console.log(tf.memory().numTensors, tf.memory().numBytes)
  return stylized;
}

process.on("unhandledRejection", (reason, promise) =>
{
  console.log('Unhandled Rejection at:', promise, 'reason:', reason);
});


var api = new ProtoPost({
  stylize: stylizeB64
});

api.start(PORT);
