lastParam_flagg = false;

function ParamsDefecte(){
  document.getElementById("minArea").value = "150";
  document.getElementById("maxArea").value = "1500";
  document.getElementById("blobColor").value = "0";
  document.getElementById("minDistBetweenBlobs").value = "10";
  document.getElementById("minThreshold").value = "100";
  document.getElementById("maxThreshold").value = "255";
  document.getElementById("thresholdStep").value = "200";
  document.getElementById("minRepeatability").value = "1";
  document.getElementById("minCircularity").value = "0";
  document.getElementById("maxCircularity").value = "0.5";
  document.getElementById("minConvexity").value = "0.4";
  document.getElementById("maxConvexity").value = "1";
  document.getElementById("minInertiaRatio").value = "0";
  document.getElementById("maxInertiaRatio").value = "0.1";
}

  
function Clean(){
  document.getElementById("minArea").value = "";
  document.getElementById("maxArea").value = "";
  document.getElementById("blobColor").value = "";
  document.getElementById("minDistBetweenBlobs").value = "";
  document.getElementById("minThreshold").value = "";
  document.getElementById("maxThreshold").value = "";
  document.getElementById("thresholdStep").value = "";
  document.getElementById("minRepeatability").value = "";
  document.getElementById("minCircularity").value = "";
  document.getElementById("maxCircularity").value = "";
  document.getElementById("minConvexity").value = "";
  document.getElementById("maxConvexity").value = "";
  document.getElementById("minInertiaRatio").value = "";
  document.getElementById("maxInertiaRatio").value = "";

}

document.getElementById('inputfile').addEventListener('change', function() { 
        
  var name = document.getElementById('inputfile');
  var nom_arxiu = name.files.item(0).name;
  var mida_arxiu = name.files.item(0).size;
  var tipus_arxiu = name.files.item(0).type;

  alert(nom_arxiu);
  alert(mida_arxiu);
  alert(tipus_arxiu);

  var fr=new FileReader(); 

  fr.onload=function(){ 
    
    document.getElementById('output').textContent=fr.result; 
  } 
  fr.readAsText(this.files[0]);   
}) 

function click(){
  alert("HERESTUFF");
  var e = document.getElementById("pr");
  e.click();

}
