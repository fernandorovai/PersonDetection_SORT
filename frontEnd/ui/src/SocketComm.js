import React from 'react';

export default class SocketComm extends React.Component {
    constructor(props) {
        super(props);
        this.state = { "detectionHistory": null }
    }

    componentDidMount() {
        var socket = new WebSocket('ws://10.114.86.79:8765');
        socket.onopen = function (e) {
            console.log("connected")
        }

        socket.onmessage = (streamData) => {
            if (typeof (streamData.data) == 'string') {
                let jsonObject = JSON.parse(streamData.data);
                // console.log(jsonObject['detectionHistory'])
                this.setState({ "detectionHistory": jsonObject['detectionHistory'] })
            }
        }
    };

}