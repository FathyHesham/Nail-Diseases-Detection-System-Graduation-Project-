$(document).ready(function () {
    // Init
    $('.image-section').hide();
    $('.loader').hide();
    $('#result').hide();

    // Upload Preview
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('#imagePreview').hide();
                $('#imagePreview').fadeIn(650);
            }
            reader.readAsDataURL(input.files[0]);
        }
    }
    $("#imageUpload").change(function () {
        $('.image-section').show();
        $('#btn-predict').show();
		$('#result').text('');
        $('#result').hide();
        readURL(this);
    });

    // Predict
    $('#btn-predict').click(function () {
        var form_data = new FormData($('#upload-file')[0]);

        // Show loading animation
        $(this).hide();
        $('.loader').show();

        // Make prediction by calling api /predict
        $.ajax({
            type: 'POST',
            url: '/predict',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            async: true,
            success: function (data) {
                // Get and display the result

                $('.loader').hide();
                $('#result').fadeIn(1);
				if (data[10] == "Healthy Nail")
					
					{$('#result').html(data[0] + "<br>" + data[1] + "<br>" + data[2] + "<br>" + data[3] + "<br>" + data[4] + "<br>" + data[5] + "<br>" + data[6] + "<br>" + data[7] + "<br>" + data[8] + "<br>" + "** Possible diseases for the symtoms is: NONE");}
				
				else if (data[10] == "Blue Finger")
				
					{$('#result').html(data[0] + "<br>" + data[1] + "<br>" + data[2] + "<br>" + data[3] + "<br>" + data[4] + "<br>" + data[5] + "<br>" + data[6] + "<br>" + data[7] + "<br>" + data[8] + "<br>" + "** Possible diseases are Asthma, Heart disease, Methhemoglobinemia, Polycythemia, Raynauds disease " + "<br>" + " I suggest that you go to a doctor specializing in 'cardiology and vascular." );}
			
				else if (data[10] == "Beaus Line")
					
					{$('#result').html(data[0] + "<br>" + data[1] + "<br>" + data[2] + "<br>" + data[3] + "<br>" + data[4] + "<br>" + data[5] + "<br>" + data[6] + "<br>" + data[7] + "<br>" + data[8] + "<br>" + "Possible diseases are zinc deficiency, pneumonia " + "<br>" + "** I suggest that you go to a doctor specializing in 'Dermatologist." );}
				
				else if (data[10] == "Clubbing")
					
					{$('#result').html(data[0] + "<br>" + data[1] + "<br>" + data[2] + "<br>" + data[3] + "<br>" + data[4] + "<br>" + data[5] + "<br>" + data[6] + "<br>" + data[7] + "<br>" + data[8] + "<br>" + " Possible diseases are Lung cancer, Lung abscess, Pulmonary tuberculosis, Congestive cardiac failure, Infective endocarditis " + "<br>" + "** I suggest that you go to a doctor specializing in ' lungs (pulmonology)" );}

				else if (data[10] == "Koilonychia")
					
					{$('#result').html(data[0] + "<br>" + data[1] + "<br>" + data[2] + "<br>" + data[3] + "<br>" + data[4] + "<br>" + data[5] + "<br>" + data[6] + "<br>" + data[7] + "<br>" + data[8] + "<br>" + "** Possible diseases are iron deficiency anemia" + "<br>" + "** I suggest that you go to a doctor specializing in 'autoimmune diseases'." );}

				else if (data[10] == "Muehrckes Lines")
					
							{$('#result').html(data[0] + "<br>" + data[1] + "<br>" + data[2] + "<br>" + data[3] + "<br>" + data[4] + "<br>" + data[5] + "<br>" + data[6] + "<br>" + data[7] + "<br>" + data[8] + "<br>" + "** Possible diseases are arsenic poisoning, thallium poisoning, hodgkins lymphoma, sickle cell, anemia cardiac failure" + "<br>" + "** I suggest that you go to a doctor specializing in 'Dermatologist'." )}
							
				else if (data[10] == "Pitting")

							{$('#result').html(data[0] + "<br>" + data[1] + "<br>" + data[2] + "<br>" + data[3] + "<br>" + data[4] + "<br>" + data[5] + "<br>" + data[6] + "<br>" + data[7] + "<br>" + data[8] + "<br>" + "** Possible diseases are Psoriasis , Rheumatoid arthritis , Eczema , Alopecia areata , Systemic lupus erythematosus , Raynaud's disease , Scleroderma" + "<br>" + "** I suggest that you go to a doctor specializing in 'Dermatologist'.");}


				else if (data[10] == "Terrys Nail")
					
							{$('#result').html(data[0] + "<br>" + data[1] + "<br>" + data[2] + "<br>" + data[3] + "<br>" + data[4] + "<br>" + data[5] + "<br>" + data[6] + "<br>" + data[7] + "<br>" + data[8] + "<br>" + "** Possible diseases are Liver disease , Kidney disease , Congestive heart failure , Diabetes" + "<br>" + "** I suggest that you go to a doctor specializing in 'endocrinologist, nutritionist or hepatologist'.");}
							
				else if (data[10] == "Acral Lentiginous Melanoma")
					
							{$('#result').html(data[0] + "<br>" + data[1] + "<br>" + data[2] + "<br>" + data[3] + "<br>" + data[4] + "<br>" + data[5] + "<br>" + data[6] + "<br>" + data[7] + "<br>" + data[8] + "<br>"	+ "** Possible diseases are Subungual hematoma , Subungual melanonychia , Warts , Onychomycosis" + "<br>" + "** I suggest that you go to a doctor specializing in 'Dermatologist'.");}
				
				else if (data[10] == "Error-Not Nail")
							{$('#result').html("Error! Please Enter nail image");}
            },
        });
    });

});
