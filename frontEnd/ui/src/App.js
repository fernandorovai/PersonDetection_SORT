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
    var socket = new WebSocket('ws://192.168.0.16:8765');
    var socket2 = new WebSocket('ws://192.168.0.16:8766');
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
        // console.log(jsonObject)

        // this.setState({ "imageSrc": jsonObject })
        // console.log(jsonObject)
      }
    }

    socket2.onmessage = (streamData) => {
      console.log(streamData)
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
