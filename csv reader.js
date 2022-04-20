// Import dependencies
const fs = require("fs");
const csv = require("csvtojson");
const { Parser } = require("json2csv");

(async () => {

    // Load the attendance
    const attendance = await csv().fromFile("attendance.csv");

    // Show the attendance
    console.log(attendance);

})();