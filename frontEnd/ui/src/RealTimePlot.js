import React from 'react';
import * as d3 from "d3";


export default class RealTimePlot extends React.Component {
    constructor(props) {
        super(props);
        this.line = null
        this.svg = null
        this.x = null
        this.y = null
        this.height = null
        this.width = null
        this.xAxis = null
        this.yAxis = null
        this.xAxisSvg = null
        this.pathsG = null
    }

    componentDidMount() {

        const margin = { top: 20, right: 20, bottom: 30, left: 40 };
        this.width = 400 - margin.left - margin.right;
        this.height = 200 - margin.top - margin.bottom;


        this.x = d3
            .scaleBand()
            .range([0, this.width])
            .padding(0.05);

        this.y = d3.scaleLinear().range([this.height, 0]);

        const container = d3
            .select('body')
            .append('div')
            .attr('class', 'container');


        const svg = container
            .append('svg')
            .attr('width', this.width + margin.left + margin.right)
            .attr('height', this.height + margin.top + margin.bottom)
            .append('g')
            .attr('transform', 'translate(' + margin.left + ',' + margin.top + ')');

        // add the x Axis
        this.xAxis = d3.axisBottom(this.x)
        this.xAxisSvg = 
            svg
            .append('g')
            .attr('transform', 'translate(0,' + this.height + ')')
            .attr('class', 'x-axis')
        this.xAxisSvg.call(this.xAxis);

        // add the y Axis
        this.yAxis = svg
            .append('g')
            .attr('class', 'y-axis')
            .call(d3.axisLeft(this.y));
        
            this.pathsG = svg.append("g").attr("id", "paths").attr("class", "paths")
        .attr("clip-path", "url(#clip2)");

        this.svg = svg

    }

    componentWillUpdate(nextProps) {
        var lineArr = []
        var parseTime = d3.timeFormat("%H:%M:%S");

        for (var i = 0; i < nextProps.detectionHistory.length; i++) {
            let el = nextProps.detectionHistory[i];
            lineArr.push({ "time": parseTime(new Date(el.datetime)), "value": el.numPerson })
        }

        var line = d3.line()
        .curve(d3.curveBasis)
        .x(function(d) {
          return this.x(d.time);
        })
        .y(function(d) {
          return this.y(d.value);
        });

        console.log("component update")
        // console.log(nextProps)
        this.x.domain(
            lineArr.map(d => {
                return d.time;
            })
        );

        this.y.domain([
            0,
            d3.max(lineArr, d => {
                return d.value;
            }),
        ]);

        
        this.svg
            .selectAll('.bar')
            .remove()
            .exit()
            .data(lineArr)
            .enter()
            .append('rect')
            .attr('class', 'bar')
            .attr('x', d => {
                return this.x(d.time);
            }).attr('width', this.x.bandwidth())
            .attr('y', d => {
                return this.y(d.value);
            }).attr('height', d => {
                return this.height - this.y(d.value);
            })

      


    //        //Join
    //   var minerG = this.pathsG.selectAll(".minerLine").data(lineArr);
    //   var minerGEnter = minerG.enter()
    //     //Enter
    //     .append("g")
    //     .attr("class", "minerLine")
    //     .merge(minerG);

    //   //Join
    //   var minerSVG = minerGEnter.selectAll("path").data(function(d) {
    //     return [d];
    //   });
    //   var minerSVGenter = minerSVG.enter()
    //     //Enter
    //     .append("path").attr("class", "line")
    //       .merge(minerSVG)
    //     //Update
    //     .transition()
    //     .duration(1)
    //     .ease(d3.easeLinear, 2)
    //     .attr("d", function(d) {
    //       return line(d)
    //     })
    //     .attr("transform", null)

        // update the x-axis
        this.xAxisSvg.transition().duration(1).ease(d3.easeLinear, 2).call(this.xAxis);

        // update the y-axis
        this.svg.select('.y-axis').call(d3.axisLeft(this.y));
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

