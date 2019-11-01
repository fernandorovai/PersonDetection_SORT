import React from 'react';
import logo from './logo.svg';
import RealTimePlot from './RealTimePlot'
import LiveBubbleChart from './BubbleChart'
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css';
import { Container, Row, Col, Navbar, Nav } from 'react-bootstrap'

class App extends React.Component {
  constructor() {
    super();
    this.state = {
      "detectionLiveSummary": null
    }
  }

  componentDidMount() {
    var socket = new WebSocket('ws://10.114.86.76:8765');
    socket.onopen = function (e) {
      console.log("connected")
    }

    socket.onmessage = (streamData) => {
      if (typeof (streamData.data) == 'string') {
        let jsonObject = JSON.parse(streamData.data);
        this.setState({ "detectionLiveSummary": jsonObject['detectionLiveSummary'], "filteredDetectionBoxes": jsonObject['filteredDetectionBoxes'] })
      }
    }
  };

  render() {
    return (<React.Fragment>
      <Navbar bg="light">
        <Navbar.Brand href="#home">MARS Café Dashboard</Navbar.Brand>
      </Navbar>
      <Container fluid>
        <Row>
          <Col className="bg-light no-gutters" md={2} >
            <Nav defaultActiveKey="/home" className="flex-column">
              <Nav.Link href="/home">Active</Nav.Link>
              <Nav.Link eventKey="link-1">Link</Nav.Link>
              <Nav.Link eventKey="link-2">Link</Nav.Link>
              <Nav.Link eventKey="disabled" disabled>
                Disabled
            </Nav.Link>
            </Nav>
          </Col>

          <Col className="bg-light no-gutters">
            <Row>
              <Col className="no-gutters lineChart" md={7}> <RealTimePlot detectionLiveSummary={this.state.detectionLiveSummary} />   </Col>
              <Col className="no-gutters liveBubbleChart" md={5}><LiveBubbleChart filteredDetectionBoxes={this.state.filteredDetectionBoxes} detectionLiveSummary={this.state.detectionLiveSummary} /></Col>
            </Row>
          </Col>
        </Row>
      </Container >
    </React.Fragment>
    )
  }
}

export default App;
