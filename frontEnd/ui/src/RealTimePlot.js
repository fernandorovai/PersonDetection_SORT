import React from 'react';
import * as d3 from "d3";
import { scale } from "d3"

export default class RealTimePlot extends React.Component {
    constructor(props) {
        super(props);

    }

    componentDidMount() {
        var svg = d3.select('body').append("svg").attr("width", 1000).attr("height", 300);
    }

    componentWillUpdate(nextProps) {
        var lineArr = []

        for (var i = 0; i < nextProps.detectionHistory.length; i++) {
            let el = nextProps.detectionHistory[i];
            lineArr.push({ "time": new Date(el.datetime), "y": el.numPerson })
        }



        console.log("component update")
        console.log(nextProps)
        var svgWidth = 500;
        var svgHeight = 300;
        var dataset = [80, 100, 56, 120, 180, 30, 40, 120, 160];

        var barPadding = 5;
        var barWidth = (svgWidth / dataset.length);

        var svg = d3.select('svg')
            .attr("width", svgWidth)
            .attr("height", svgHeight)

        var barChart = svg.selectAll("rect")
            .datum(lineArr)
            .enter()
            .append("rect")
            .attr("y", function (d) {
                return svgHeight - d
            })
            .attr("height", function (d) {
                return d;
            })
            .attr("width", barWidth - barPadding)
            .attr("transform", function (d, i) {
                var translate = [barWidth * i, 0];
                return "translate(" + translate + ")";
            }).call(chart)
    }

    render() {
        return <div className="realTimeChart"></div>
    }

    //render
    // render() {
    //     console.log(this.props.detectionHistory)
    //     console.log("Rendering chart")
    //     return (<div><div>Render realtime chart</div>
    //         <div>{this.props.test}</div></div>)
    // }
}

