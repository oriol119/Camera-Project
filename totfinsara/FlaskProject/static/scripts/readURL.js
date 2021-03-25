function readURL(input) {
    if (input.files && input.files[0]) { //Revisamos que el input tenga contenido
        var reader = new FileReader(); //Leemos el contenido
        
        var nom_arxiu = input.files.item(0).name;
        var mida_arxiu = input.files.item(0).size;
        var tipus_arxiu = input.files.item(0).type;
        document.getElementById("nom_frame").value = nom_arxiu;
        
        reader.onload = function(e) { //Al cargar el contenido lo pasamos como atributo de la imagen de arriba
        $('#blah').attr('src', e.target.result);
        }
        
        reader.readAsDataURL(input.files[0]);
    }
}

$("#imgInp").change(function() { //Cuando el input cambie (se cargue un nuevo archivo) se va a ejecutar de nuevo el cambio de imagen y se verá reflejado.
readURL(this);
});

function readURL2(input) {
    if (input.files && input.files[0]) { //Revisamos que el input tenga contenido
        var reader = new FileReader(); //Leemos el contenido
        
        var nom_arxiu = input.files.item(0).name;
        var mida_arxiu = input.files.item(0).size;
        var tipus_arxiu = input.files.item(0).type;
        document.getElementById("nom_frame2").value = nom_arxiu;
        
        reader.onload = function(e) { //Al cargar el contenido lo pasamos como atributo de la imagen de arriba
        $('#blah2').attr('src', e.target.result);
        }
        
        reader.readAsDataURL(input.files[0]);
    }
}

$("#imgInp2").change(function() { //Cuando el input cambie (se cargue un nuevo archivo) se va a ejecutar de nuevo el cambio de imagen y se verá reflejado.
readURL2(this);
});