import React from 'react';
import * as d3 from "d3";

export default class LiveBubbleChart extends React.Component {
    constructor(props) {
        super(props);
        this.bubbleChart = null;
        this.width = 400;
        this.height = 400;
        this.center = { x: this.width / 2, y: this.height / 2 };
        this.svg = null;
        this.bubbles = null;
        this.nodes = [];
        this.force = null;
        this.forceStrength = 0.1;
        this.simulation = null;
        this.elements = null;
        this.force = null;
    }

    // Charge function that is called for each node.
    // Charge is proportional to the diameter of the
    // circle (which is stored in the radius attribute
    // of the circle's associated data.
    // This is done to allow for accurate collision
    // detection with nodes of different sizes.
    // Charge is negative because we want nodes to repel.
    // Dividing by 8 scales down the charge to be
    // appropriate for the visualization dimensions.
    charge = (d) => {
        return -Math.pow(d.radius, 2.0) * this.forceStrength;
    }

    tick = () => {
        if (this.bubbles) {
            this.bubbles.attr('cx', d => d.x).attr('cy', d => d.y)
        }
    }

    /*
      * This data manipulation function takes the raw data from
      * the socket and converts it into an array of node objects.
      * Each node will store data and visualization values to visualize
      * a bubble.
      *
      * rawData is expected to be an array of data objects, read in from
      * one of d3's loading functions like d3.csv.
      *
      * This function returns the new node array, with a node in that
      * array for each element in the rawData input.
      */
    createNodes(rawData) {
        const maxSize = d3.max(rawData, d => +d.aliveTime);

        // Sizes bubbles based on their area instead of raw radius
        const radiusScale = d3.scaleSqrt()
            .domain([0, maxSize])
            .range([0, 50])

        // Use map() to convert raw data into node data.
        // Checkout http://learnjsdata.com/ for more on
        // working with data.
        // use map() to convert raw data into node data
        const myNodes = rawData.map(d => ({
            id: d.boxID,
            radius: radiusScale(+d.aliveTime),
            size: +d.aliveTime,
            x: d.x,
            y: d.y
        }))

        // sort them to prevent occlusion of smaller nodes.
        myNodes.sort(function (a, b) { return b.size - a.size; });

        return myNodes;
    }


    componentDidMount = () => {
        // Create a SVG element inside the provided selector
        // with desired size.
        this.svg = d3.select('.liveBubbleChart')
            .append('svg')
            .attr('width', this.width)
            .attr('height', this.height);

        var force = d3.forceSimulation()
            .velocityDecay(0.2)
            .force('charge', d3.forceManyBody().strength(this.charge))
            .force('center', d3.forceCenter(this.center.x, this.center.y))
            .force('x', d3.forceX().strength(this.forceStrength).x(this.center.x))
            .force('y', d3.forceY().strength(this.forceStrength).y(this.center.y))
            .force('collision', d3.forceCollide().radius(d => d.radius + 1))
            .on('tick', this.tick);
    }

    componentWillUpdate = (nextProps) => {
        console.log(nextProps.filteredDetectionBoxes)

        let nodes = this.createNodes(nextProps.filteredDetectionBoxes);
        this.bubbles = this.svg.selectAll('circle')
            .data(nodes, function (d) { return d.id; })
            .join('circle')
            .classed('circle', true)
            .attr('r', d => d.radius)
        var force = d3.forceSimulation()
            .velocityDecay(0.2)
            .force('charge', d3.forceManyBody().strength(this.charge))
            .force('center', d3.forceCenter(this.center.x, this.center.y))
            .force('x', d3.forceX().strength(this.forceStrength).x(this.center.x))
            .force('y', d3.forceY().strength(this.forceStrength).y(this.center.y))
            .force('collision', d3.forceCollide().radius(d => d.radius + 1))
            .on('tick', this.tick);

        force.nodes(nodes)
        force.restart();
    }


    render() {
        return (null);
    }
}
