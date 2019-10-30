import React from 'react';
import logo from './logo.svg';
import './App.css';

class App extends React.Component {
  constructor() {
    super();
    this.state = {
      "imageSrc": null
    }
  }

  componentDidMount() {
    var socket = new WebSocket('ws://10.114.86.79:8765');
    socket.onopen = function (e) {
      console.log("connected")
    }

    let imageUrl;
    socket.onmessage = (streamData) => {
      if (typeof (streamData.data) == 'object') {
        // console.log(streamData.data)
        var blob = streamData.data;

        // var reader2 = new FileReader();
        // reader2.onload = function(e) {
        //     console.log(e.target.result)
        // }
        // reader2.readAsArrayBuffer(blob)

        // reader2.readAsText(blob);

        var reader = new FileReader();
        reader.onload = function (e) {
          imageUrl = e.target.result;
        }
        reader.readAsDataURL(blob);
        this.setState({ "imageSrc": imageUrl })

      }
      else if (typeof (streamData.data) == 'string') {

        let jsonObject = JSON.parse(streamData.data);
        console.log(jsonObject['detectionHistory'])

        // this.setState({ "imageSrc": jsonObject })
        // console.log(jsonObject)
      }
    }


  };

  render() {
    return (
      <div>
        <canvas ref="canvas" width={640} height={425} />
        <img ref="image" src={this.state.imageSrc} />
      </div>
    )
  }
}

export default App;
