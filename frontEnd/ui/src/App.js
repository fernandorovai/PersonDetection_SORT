import React from 'react';
import logo from './logo.svg';
import './App.css';
import RealTimePlot from './RealTimePlot'
import LiveBubbleChart from './BubbleChart'

class App extends React.Component {
  constructor() {
    super();
    this.state = {
      "detectionLiveSummary": null
    }
  }

  componentDidMount() {
    var socket = new WebSocket('ws://192.168.0.16:8765');
    socket.onopen = function (e) {
      console.log("connected")
    }

    let imageUrl;
    socket.onmessage = (streamData) => {
      if (typeof (streamData.data) == 'string') {
        let jsonObject = JSON.parse(streamData.data);
        this.setState({ "detectionLiveSummary": jsonObject['detectionLiveSummary'], "filteredDetectionBoxes": jsonObject['filteredDetectionBoxes'] })
      }
    }
  };

  render() {
    return (
      <div>
        <div className="realTimeChart"><RealTimePlot detectionLiveSummary={this.state.detectionLiveSummary} /> </div>
        <div className="liveBubbleChart"><LiveBubbleChart filteredDetectionBoxes={this.state.filteredDetectionBoxes} /></div>
      </div>)
  }
}

export default App;
