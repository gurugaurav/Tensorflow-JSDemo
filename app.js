
var coords = [];
var m_down = false;
var canvas;
var classNames = [];
var model;



$( document ).ready(function() {
    canvas = window._canvas = new fabric.Canvas('canvas');
    canvas.backgroundColor = '#ffffff';
    canvas.isDrawingMode = 0;
    canvas.freeDrawingBrush.color = "black";
    canvas.freeDrawingBrush.width = 10;
    canvas.renderAll();
    //setup listeners 
    canvas.on('mouse:up', function(e) {
        getFrame();
        m_down = false
    });
    canvas.on('mouse:down', function(e) {
        m_down = true
    });
    canvas.on('mouse:move', function(e) {
        recordCoor(e)
    });
})


function setTable(top_class, probs) {
    for (var i = 0; i < top_class.length; i++) {
        let sym = document.getElementById('pred_class' + (i + 1))
        let prob = document.getElementById('prob' + (i + 1))
        sym.innerHTML = top_class[i]
        prob.innerHTML = Math.round(probs[i] * 100)
    }
  
    createPie(".pieID.legend", ".pieID.pie");

}


function recordCoor(event) {
    var pointer = canvas.getPointer(event.e);
    var posX = pointer.x;
    var posY = pointer.y;

    if (posX >= 0 && posY >= 0 && m_down) {
        coords.push(pointer)
    }
}


function getMinBox() {
    var coorX = coords.map(function(p) {
        return p.x
    });
    var coorY = coords.map(function(p) {
        return p.y
    });

    //find top left and bottom right corners 
    var min_coords = {
        x: Math.min.apply(null, coorX),
        y: Math.min.apply(null, coorY)
    }
    var max_coords = {
        x: Math.max.apply(null, coorX),
        y: Math.max.apply(null, coorY)
    }

    return {
        min: min_coords,
        max: max_coords
    }
}


function getImageData() {
        const mbb = getMinBox()

        
        const dpi = window.devicePixelRatio
        const imgData = canvas.contextContainer.getImageData(mbb.min.x * dpi, mbb.min.y * dpi,
                                                      (mbb.max.x - mbb.min.x) * dpi, (mbb.max.y - mbb.min.y) * dpi);
        return imgData
    }


function getFrame() {
    //make sure we have at least two recorded coordinates 
    if (coords.length >= 2) {

        //get the image data from the canvas 
        const imgData = getImageData()

        //get the prediction 
        const pred = model.predict(preprocess(imgData)).dataSync()

        //find the top 5 predictions 
        const indices = findIndicesOfMax(pred, 5)
        const probs = findTopValues(pred, 5)
        const names = getClassNames(indices)

        //set the table 
        setTable(names, probs)
    }

}


function getClassNames(indices) {
    var outp = []
    for (var i = 0; i < indices.length; i++)
        outp[i] = classNames[indices[i]]
    return outp
}


function loadDict() {
    $.ajax({
        url: 'http://localhost:8000/AICanvas/model/classes.txt',
        dataType: 'text',
       // data: data,
        success : function( data, textStatus, jqXHR ) {
            const lst = data.split(/\n/)
            for (var i = 0; i < lst.length - 1; i++) {
                let symbol = lst[i]
                classNames[i] = symbol
            }
        },
        error: function(x) { 
            console.log(data);  
        }
    });
}


function success(data) {
   
}


function findIndicesOfMax(inp, count) {
    var outp = [];
    for (var i = 0; i < inp.length; i++) {
        outp.push(i); // add index to output array
        if (outp.length > count) {
            outp.sort(function(a, b) {
                return inp[b] - inp[a];
            }); // descending sort the output array
            outp.pop(); // remove the last index (index of smallest element in output array)
        }
    }
    return outp;
}


function findTopValues(inp, count) {
    var outp = [];
    let indices = findIndicesOfMax(inp, count)
    // show 5 greatest scores
    for (var i = 0; i < indices.length; i++)
        outp[i] = inp[indices[i]]
    return outp
}


function preprocess(imgData) {
    return tf.tidy(() => {
        //convert to a tensor 
        let tensor = tf.fromPixels(imgData, numChannels = 1)
        
        //resize 
        const resized = tf.image.resizeBilinear(tensor, [28, 28]).toFloat()
        
        //normalize 
        const offset = tf.scalar(255.0);
        const normalized = tf.scalar(1.0).sub(resized.div(offset));

        //We add a dimension to get a batch shape 
        const batched = normalized.expandDims(0)
        return batched
    })
}


async function run() {
   
    model = await tf.loadModel('http://localhost:8000/AICanvas/model/model.json')
    
    
    model.predict(tf.zeros([1, 28, 28, 1]))
    
   
    allowDrawing()
    
   
    await loadDict()
}


function allowDrawing() {
    canvas.isDrawingMode = 1;
    document.getElementById('status').innerHTML = 'Model Loaded';
    $('button').prop('disabled', false);
    // var slider = document.getElementById('myRange');
    // slider.oninput = function() {
    //     canvas.freeDrawingBrush.width = this.value;
   // };
}


function erase() {
    canvas.clear();
    canvas.backgroundColor = '#ffffff';
    coords = [];
}


run();