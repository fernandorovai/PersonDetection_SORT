import React from 'react';
import { Row, Col, } from 'react-bootstrap'

export default class LiveSummaryBar extends React.Component {
    constructor(props) {
        super(props);
    }


    render() {
        let numLiveDetections = 0
        let maxTimeAlive = 0
        let avgStayTime = 0;

        if (this.props.numLiveDetections)
            numLiveDetections = this.props.numLiveDetections;

        if (this.props.maxTimeAlive)
            maxTimeAlive = this.props.maxTimeAlive > 0 ? (this.props.maxTimeAlive / 60).toFixed(2) : 0;

        if (this.props.avgStayTime)
            avgStayTime = this.props.avgStayTime;

        return (<Row>
            <Col md={3}>
                <h1 className="text-success">{numLiveDetections}</h1>
                <span className="smallTitle text-success">Person Now</span>
            </Col>
            <Col>
                <h1>{avgStayTime}</h1>
                <span className="smallTitle">Avg. stay time (min)</span>
            </Col>
            <Col>
                <h1>{maxTimeAlive}</h1>
                <span className="smallTitle ">Max. stay time (min)</span>
            </Col>

        </Row>)
    }


}
