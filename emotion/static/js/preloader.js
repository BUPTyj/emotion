//Preloader
  loader();
  function loader() {
      var blueket = document.querySelector('.preloader'),
      inner = document.querySelector('.preloader_inner');    
      var w = 0,
          t = setInterval(function() {
              w = w + 1;
              inner.textContent = w+ '%';
              if (w === 100){
                  blueket.classList.add('page-loaded');        
                  clearInterval(t);
                  w = 0;               
              }
      }, 20);
  }