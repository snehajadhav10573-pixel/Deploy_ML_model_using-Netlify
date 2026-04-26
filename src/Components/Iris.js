import { useState, useEffect } from "react";

export function useIrisData() {
  const [data, setData] = useState([]);

  useEffect(() => {
    fetch("https://raw.githubusercontent.com/uiuc-cse/data-fa14/gh-pages/data/iris.csv")
      .then((res) => res.text())
      .then((text) => {
        const rows = text.trim().split("\n");
        const headers = rows[0].split(",");
        const jsonData = rows.slice(1).map((row) => {
          const values = row.split(",");
          let obj = {};
          headers.forEach((header, index) => {
            obj[header] = values[index];
          });
          return obj;
        });
        setData(jsonData);
      })
      .catch((err) => console.error(err));
  }, []);

  // Return just the data
  return data;
}

export default useIrisData;