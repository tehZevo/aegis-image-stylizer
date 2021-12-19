require("@tensorflow/tfjs-node");
var tf = require("@tensorflow/tfjs");
var ProtoPost = require("protopost");
var utils = require("./utils");

//TODO: env vars
var PORT = 3001;
var STYLE_NAME = "mobilenet-style";
var TRANSFORMER_NAME = "seperable-transformer";
var STYLE_RATIO = 0.8; //0..1
var SIZE = 512;

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
  styles = await Promise.all(styles.map((e) => utils.loadImageB64(e, SIZE/2))); //TODO: remove hardcoded scale
  styles = await Promise.all(styles.map((e) => getStyle(e)));

  var sourceStyle = await getStyle(content);
  var targetStyle = await combineStyles(styles);

  var style = await combineStyles([sourceStyle, targetStyle], [0.2, 0.8]); //TODO: remove hardcoded style

  var stylized = await stylize(content, style);

  stylized = await utils.saveImageB64(stylized);

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
