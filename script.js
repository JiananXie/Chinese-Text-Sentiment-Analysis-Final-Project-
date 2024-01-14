function getSentiment() {
    var text = document.getElementById("myText").value;
    var select = document.getElementById("classificationType");
    var classificationType = select.options[select.selectedIndex].value;

    fetch('http://localhost:5000/sentiment', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({text: text, classificationType: classificationType}),
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            console.error('Server error:', data.error);
            document.getElementById("result").innerHTML = 'Error: ' + data.error;
        } else {
            document.getElementById("result").innerHTML = data.result;
        }
    })
    .catch((error) => {
        console.error('Error:', error);
    });
}