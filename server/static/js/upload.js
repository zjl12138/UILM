// Constants
var MAX_UPLOAD_FILE_SIZE = 1024*1024*50; // 50 MB
var UPLOAD_URL = "/new_get_bbox";
var NEXT_URL   = "/show/";

// pending files to handle when the Upload button is finally clicked.
var PENDING_FILE  = null;


$(document).ready(function() {
    // Set up the drag/drop zone.
    initDropbox();

    // Set up the handler for the file input box.
    $("#file-picker").on("change", function() {
        handleFiles(this.files);
    });

    // Handle the submit button.
    $("#upload-button").on("click", function(e) {
        // If the user has JS disabled, none of this code is running but the
        // file upload input box should still work. In this case they'll
        // just POST to the upload endpoint directly. However, with JS we'll do
        // the POST using ajax and then redirect them ourself when done.
        e.preventDefault();
        doUpload();
    })
});


function doUpload() {
    $("#progress").show();
    var $progressBar   = $("#progress-bar");

    // Gray out the form.
    $("#upload-form :input").attr("disabled", "disabled");

    // Initialize the progress bar.
    $progressBar.css({"width": "0%"});

    // Collect the form data.
    fd = collectFormData();

    fd.append("file", PENDING_FILE);

    // Inform the back-end that we're doing this over ajax.
    fd.append("__ajax", "true");

    var xhr = $.ajax({
        xhr: function() {
            var xhrobj = $.ajaxSettings.xhr();
            if (xhrobj.upload) {
                xhrobj.upload.addEventListener("progress", function(event) {
                    var percent = 0;
                    var position = event.loaded || event.position;
                    var total    = event.total;
                    if (event.lengthComputable) {
                        percent = Math.ceil(position / total * 100);
                    }

                    // Set the progress bar.
                    $progressBar.css({"width": percent + "%"});
                    $progressBar.text(percent + "%");
                }, false)
            }
            return xhrobj;
        },
        url: UPLOAD_URL,
        method: "POST",
        contentType: false,
        processData: false,
        cache: false,
        data: fd,
        success: function(data) {
            $progressBar.css({"width": "100%"});
            data = JSON.parse(data);

            // How'd it go?
            if (data.status === "error") {
                // Uh-oh.
                window.alert(data.msg);
                $("#upload-form :input").removeAttr("disabled");
                return;
            }
            else {
                // Ok! Get the UUID.
                var image_path = data.msg;
                window.location = NEXT_URL + image_path;
            }
        },
    });
}


function collectFormData() {
    // Go through all the form fields and collect their names/values.
    var fd = new FormData();

    $("#upload-form :input").each(function() {
        var $this = $(this);
        var name  = $this.attr("name");
        var type  = $this.attr("type") || "";
        var value = $this.val();

        // No name = no care.
        if (name === undefined) {
            return;
        }

        // Skip the file upload box for now.
        if (type === "file") {
            return;
        }

        // Checkboxes? Only add their value if they're checked.
        if (type === "checkbox" || type === "radio") {
            if (!$this.is(":checked")) {
                return;
            }
        }

        fd.append(name, value);
    });

    return fd;
}


function handleFiles(files) {
    PENDING_FILE = files[0]
}


function initDropbox() {
    var $dropbox = $("#dropbox");

    // On drag enter...
    $dropbox.on("dragenter", function(e) {
        e.stopPropagation();
        e.preventDefault();
        $(this).addClass("active");
    });

    // On drag over...
    $dropbox.on("dragover", function(e) {
        e.stopPropagation();
        e.preventDefault();
    });

    // On drop...
    $dropbox.on("drop", function(e) {
        e.preventDefault();
        $(this).removeClass("active");

        // Get the files.
        var files = e.originalEvent.dataTransfer.files;
        handleFiles(files);

        // Update the display to acknowledge the number of pending files.
        $dropbox.text(PENDING_FILES.length + " files ready for upload!");
    });

    // If the files are dropped outside of the drop zone, the browser will
    // redirect to show the files in the window. To avoid that we can prevent
    // the 'drop' event on the document.
    function stopDefault(e) {
        e.stopPropagation();
        e.preventDefault();
    }
    $(document).on("dragenter", stopDefault);
    $(document).on("dragover", stopDefault);
    $(document).on("drop", stopDefault);
}

function jq_ChainCombo(el) {
    var selected = $(el).find(':selected').data('id'); // get parent selected options' data-id attribute
    
    // get next combo (data-nextcombo attribute on parent select)
    var next_combo = $(el).data('nextcombo');
    
    // now if this 2nd combo doesn't have the old options list stored in it, make it happen
    if(!$(next_combo).data('store'))
        $(next_combo).data('store', $(next_combo).find('option')); // store data

    // now include data stored in attribute for use...
    var options2 = $(next_combo).data('store');

    // update combo box with filtered results
    $(next_combo).empty().append(
        options2.filter(function(){
            return $(this).data('option') === selected;
        })
    );
    
    // now enable in case disabled... 
    $(next_combo).prop('disabled', false);

    // now if this combo box has a child combo box, run this function again (recursive until an end is reached)
    if($(next_combo).data('nextcombo') !== undefined )
        jq_ChainCombo(next_combo); // now next_combo is the defining combo
}

jQuery.fn.chainCombo = function() {
    // find all divs with a data-nextcombo attribute
    $('[data-nextcombo]').each(function(i, obj) {
        $(this).change(function (){
            jq_ChainCombo(this);
        });
    });
}();

jq_ChainCombo($('.select-model'))