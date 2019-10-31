import React from 'react';
import logo from './logo.svg';
import './App.css';
import RealTimePlot from './RealTimePlot'

class App extends React.Component {
  constructor() {
    super();
    this.state = {
      "detectionHistory": null
    }
  }

  componentDidMount() {
    var socket = new WebSocket('ws://0.0.0.0:8765');
    socket.onopen = function (e) {
      console.log("connected")
    }

    let imageUrl;
    socket.onmessage = (streamData) => {
      if (typeof (streamData.data) == 'string') {
        let jsonObject = JSON.parse(streamData.data);
        this.setState({ "detectionHistory": jsonObject['detectionHistory'] })
      }
    }
  };

  render() {
    return (<div className="realTimeChart"><RealTimePlot detectionHistory={this.state.detectionHistory} /></div>)
  }
}

export default App;
